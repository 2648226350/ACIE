import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from scipy.stats import percentileofscore


class NeuronLifecycleManager:
    def __init__(self, model, plasticity_factor=1, protection_factor=1,
                 reset_threshold=0.15, protect_threshold=0.3,
                 plasticity_decay=1, update_interval=1, freeze_protected=False):
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

        self.register_hooks()

        self.protected_neurons = set()
        self.reset_history = defaultdict(int)
        self.epoch_count = 0
        self.freeze_protected = freeze_protected
        self.protected_neurons_params = {}  # 存储保护神经元的参数标识



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

    def apply_protection(self, optimizer):
        """应用保护机制：修改优化器的梯度或参数更新行为"""
        if not self.protected_neurons_params:
            return

        if self.freeze_protected:
            # 方法1：冻结保护参数
            self._freeze_protected_params(optimizer)
        else:
            # 方法2：梯度屏蔽
            self._zero_protected_gradients()

    def _freeze_protected_params(self, optimizer):
        """冻结保护神经元的参数"""
        # 首先解冻所有参数
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                param.requires_grad = True

        # 冻结保护参数
        for name, module in self.model.named_modules():
            if name not in self.protected_neurons_params:
                continue

            protected_idxs = self.protected_neurons_params[name]

            if isinstance(module, nn.Conv2d):
                # 冻结卷积层特定通道
                for idx in protected_idxs:
                    if idx < module.weight.size(0):  # 确保索引有效
                        module.weight[idx].requires_grad = False
                        if module.bias is not None:
                            module.bias[idx].requires_grad = False

            elif isinstance(module, nn.Linear):
                # 冻结全连接层特定神经元
                for idx in protected_idxs:
                    if idx < module.weight.size(0):  # 确保索引有效
                        module.weight[idx].requires_grad = False
                        if module.bias is not None:
                            module.bias[idx].requires_grad = False

    def _zero_protected_gradients(self):
        """将保护神经元的梯度置零"""
        for name, module in self.model.named_modules():
            if name not in self.protected_neurons_params:
                continue

            protected_idxs = self.protected_neurons_params[name]

            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                # 卷积层梯度置零
                for idx in protected_idxs:
                    if idx < module.weight.size(0):
                        module.weight.grad[idx] = 0.0

                if module.bias is not None and module.bias.grad is not None:
                    for idx in protected_idxs:
                        if idx < module.bias.size(0):
                            module.bias.grad[idx] = 0.0

            elif isinstance(module, nn.Linear) and module.weight.grad is not None:
                # 全连接层梯度置零
                for idx in protected_idxs:
                    if idx < module.weight.size(0):
                        module.weight.grad[idx] = 0.0

                if module.bias is not None and module.bias.grad is not None:
                    for idx in protected_idxs:
                        if idx < module.bias.size(0):
                            module.bias.grad[idx] = 0.0
    def adaptive_learning_rate(self, base_lr, contribution_metrics):
        """根据神经元贡献度调整学习率"""
        lr_factors = {}

        for name, contribution in contribution_metrics.items():
            module = dict(self.model.named_modules()).get(name)
            if not module:
                continue

            # 标准化贡献度 [0, 1]
            contrib_norm = (contribution - contribution.min()) / \
                           (contribution.max() - contribution.min() + 1e-10)

            # 计算学习率因子：贡献度高的神经元学习率低
            factors = np.ones_like(contrib_norm)
            factors[contrib_norm > self.protect_threshold] = 0.1  # 高贡献神经元降低学习率
            factors[contrib_norm < self.reset_threshold] = 1.5  # 低贡献神经元提高学习率

            lr_factors[name] = factors

        return lr_factors

    def _calculate_metrics(self, test_len=0):
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
                # print(activations, gradients)
                contribution = (activations[:len(activations)-test_len] * gradients).mean(dim=0).abs().cpu().numpy()
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
            activation_norm = (activation - activation.min()) / (activation.max() - activation.min() + (activation.min()/100))
            contribution_norm = (contribution - contribution.min()) / (contribution.max() - contribution.min() + (contribution.min()/100))

            # 计算综合生命周期分数
            life_score = (self.plasticity_factor * (1 - activation_norm) +
                          self.protection_factor * contribution_norm)

            for i in range(n_neurons):
                neuron_id = f"{name}__{i}"

                # 保护决策
                if contribution_norm[i] > effective_protect_threshold or neuron_id in self.protected_neurons:
                    protect_candidates.append(neuron_id)

                # 重置决策
                # else:
                #     reset_candidates.append((name, i))
                elif activation_norm[i] < effective_reset_threshold and life_score[i] > 0.5:  # 平衡条件
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
        # 识别并记录保护神经元的参数
        self.protected_neurons.update(protect_candidates)

        # 清除旧记录
        self.protected_neurons_params.clear()

        for neuron_id in self.protected_neurons:
            parts = neuron_id.split('__')
            if len(parts) < 2:
                continue

            layer_name = '__'.join(parts[:-1])
            neuron_idx = int(parts[-1])

            module = dict(self.model.named_modules()).get(layer_name)
            if not module:
                continue

            # 存储参数位置信息
            if layer_name not in self.protected_neurons_params:
                self.protected_neurons_params[layer_name] = set()
            self.protected_neurons_params[layer_name].add(neuron_idx)


    def update_plasticity(self):
        """更新可塑性因子"""
        self.plasticity_factor *= self.plasticity_decay

    def step(self, reset=True, test_len=0):
        """执行神经元生命周期管理"""
        self.epoch_count += 1

        # 间隔更新或首次更新
        if self.epoch_count % self.update_interval != 0 and self.epoch_count >= 1:
            self.activation_records.clear()
            self.gradient_records.clear()
            return

        # 计算指标
        activation_metrics, contribution_metrics = self._calculate_metrics(test_len=test_len)

        # 决策神经元操作
        reset_candidates, protect_candidates = self._decide_neuron_actions(
            activation_metrics, contribution_metrics
        )

        # 执行操作
        if reset:
            self._reset_neurons(reset_candidates)
        self._protect_neurons(protect_candidates)

        # 更新可塑性
        self.update_plasticity()

        # 重置记录
        self.activation_records.clear()
        self.gradient_records.clear()

        # 打印统计
        print(f"Epoch {self.epoch_count}: Reset {len(reset_candidates)} neurons, "
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

    def register_hooks(self):
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

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []