import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from scipy.stats import percentileofscore


class NeuronLifecycleManager:
    def __init__(self, model, plasticity_factor=0.3, protection_factor=0.7,
                 reset_threshold=0.2, protect_threshold=0.8,
                 plasticity_decay=0.95, update_interval=5):
        """
        动态神经元生命周期管理器

        参数:
        plasticity_factor (float): 可塑性权重 (0-1), 越高越倾向重置神经元
        protection_factor (float): 保护权重 (0-1), 越高越倾向保护重要神经元
        reset_threshold (float): 重置阈值 (0-1), 低于此值考虑重置
        protect_threshold (float): 保护阈值 (0-1), 高于此值强制保护
        plasticity_decay (float): 可塑性衰减因子, 随训练降低重置概率
        update_interval (int): 更新间隔 (epochs)
        """
        self.model = model
        self.plasticity_factor = plasticity_factor
        self.protection_factor = protection_factor
        self.reset_threshold = reset_threshold
        self.protect_threshold = protect_threshold
        self.plasticity_decay = plasticity_decay
        self.update_interval = update_interval

        # 注册钩子捕获激活和梯度
        self.activation_records = defaultdict(list)
        self.gradient_records = defaultdict(list)
        self.hooks = []

        # 注册前向和反向钩子
        for m_name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 前向钩子
                hook_fw = module.register_forward_hook(
                    lambda m, inp, out, name=m_name:
                    self._forward_hook(m, inp, out, name)
                )
                # 反向钩子
                hook_bw = module.register_full_backward_hook(
                    lambda m, grad_inp, grad_out, name=m_name:
                    self._backward_hook(m, grad_inp, grad_out, name)
                )
                self.hooks.extend([hook_fw, hook_bw])

        self.protected_neurons = set()
        self.reset_history = defaultdict(int)
        self.epoch_count = 0

    def _forward_hook(self, module, inputs, outputs, name):
        """捕获前向传播激活值"""
        if isinstance(module, nn.Conv2d):
            # 卷积层: 计算通道平均激活 (batch, C, H, W) -> (C,)
            channel_avg = outputs.detach().mean(dim=(0, 2, 3))
            self.activation_records[name].append(channel_avg)
        elif isinstance(module, nn.Linear):
            # 全连接层: 直接记录输出 (batch, D) -> (D,)
            self.activation_records[name].append(outputs.detach().mean(dim=0))

    def _backward_hook(self, module, grad_input, grad_output, name):
        """捕获反向传播梯度"""
        if isinstance(module, nn.Conv2d):
            # 卷积层梯度: (batch, C, H, W)
            if grad_output[0] is not None:
                grad = grad_output[0].detach()
                channel_grad = grad.mean(dim=(0, 2, 3))
                self.gradient_records[name].append(channel_grad)
        elif isinstance(module, nn.Linear):
            # 全连接层梯度: (batch, D)
            if grad_output[0] is not None:
                self.gradient_records[name].append(grad_output[0].detach().mean(dim=0))

    def _calculate_metrics(self, test_act=4):
        """计算神经元活跃度和贡献度指标"""
        activation_metrics = {}
        contribution_metrics = {}

        for name in self.activation_records:
            # 计算平均激活强度
            activations = torch.stack(self.activation_records[name])
            activation_strength = activations.abs().mean(dim=0).cpu().numpy()

            # 计算贡献度 (激活 * 梯度)
            if name in self.gradient_records and len(self.gradient_records[name]) > 0:
                gradients = torch.stack(self.gradient_records[name])
                contribution = (activations[:len(activations)-test_act] * gradients).mean(dim=0).abs().cpu().numpy()
            else:
                contribution = np.zeros_like(activation_strength)

            activation_metrics[name] = activation_strength
            contribution_metrics[name] = contribution

        return activation_metrics, contribution_metrics

    def _decide_neuron_actions(self, activation_metrics, contribution_metrics):
        """决定神经元操作: 重置或保护"""
        reset_candidates = []
        protect_candidates = []

        # 动态调整阈值
        effective_reset_threshold = self.reset_threshold * self.plasticity_factor
        effective_protect_threshold = self.protect_threshold * self.protection_factor

        for name, activation in activation_metrics.items():
            contribution = contribution_metrics[name]
            n_neurons = len(activation)

            # 标准化指标 (层内)
            activation_norm = (activation - activation.min()) / (activation.max() - activation.min() + 1e-10)
            contribution_norm = (contribution - contribution.min()) / (contribution.max() - contribution.min() + 1e-10)

            # 计算综合生命周期分数
            life_score = (self.plasticity_factor * (1 - activation_norm) +
                          self.protection_factor * contribution_norm)

            for i in range(n_neurons):
                neuron_id = f"{name}__{i}"

                # 保护决策
                if (contribution_norm[i] > effective_protect_threshold or neuron_id in self.protected_neurons):
                    protect_candidates.append(neuron_id)

                # 重置决策
                elif (activation_norm[i] < effective_reset_threshold and
                      life_score[i] > 0.5):  # 平衡条件
                    reset_candidates.append((name, i))

        return reset_candidates, protect_candidates

    def _reset_neurons(self, reset_candidates):
        """重置选定神经元"""
        for name, neuron_idx in reset_candidates:
            module = dict(self.model.named_modules())[name]

            if isinstance(module, nn.Conv2d):
                # 重置卷积核通道
                nn.init.kaiming_normal_(module.weight[neuron_idx])
                if module.bias is not None:
                    nn.init.zeros_(module.bias[neuron_idx])

            elif isinstance(module, nn.Linear):
                # 重置全连接神经元
                nn.init.kaiming_normal_(module.weight[neuron_idx:neuron_idx + 1])
                if module.bias is not None:
                    nn.init.zeros_(module.bias[neuron_idx])

            self.reset_history[f"{name}__{neuron_idx}"] += 1

    def _protect_neurons(self, protect_candidates):
        """保护重要神经元"""
        self.protected_neurons.update(protect_candidates)

    def update_plasticity(self):
        """更新可塑性因子"""
        self.plasticity_factor *= self.plasticity_decay

    def step(self, test_act=4):
        """执行神经元生命周期管理"""
        self.epoch_count += 1

        # 间隔更新或首次更新
        if self.epoch_count % self.update_interval != 0 and self.epoch_count > 1:
            self.activation_records.clear()
            self.gradient_records.clear()
            return

        # 计算指标
        activation_metrics, contribution_metrics = self._calculate_metrics(test_act)

        # 决策神经元操作
        reset_candidates, protect_candidates = self._decide_neuron_actions(
            activation_metrics, contribution_metrics
        )

        # 执行操作
        self._reset_neurons(reset_candidates)
        self._protect_neurons(protect_candidates)

        # 更新可塑性
        self.update_plasticity()

        # 重置记录
        self.activation_records.clear()
        self.gradient_records.clear()

        # 打印统计
        print(f"Task {self.epoch_count}: Reset {len(reset_candidates)} neurons, "
              f"Protected {len(protect_candidates)} neurons, "
              f"Plasticity: {self.plasticity_factor:.4f}")

    def get_status(self):
        """获取当前神经元状态"""
        return {
            'plasticity_factor': self.plasticity_factor,
            'protected_count': len(self.protected_neurons),
            'total_resets': sum(self.reset_history.values()),
            'reset_distribution': dict(self.reset_history)
        }

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []