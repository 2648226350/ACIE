import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore


class MOO_NeuronLifecycleManager:
    """
    基于多目标优化(NSGA-II)的神经元生命周期管理器
    创新点：使用Pareto最优解决策神经元重置，平衡可塑性与稳定性
    """
    # imagenet alpha_init=0.003,beta_init=1.0,lambda_adapt=0.02,mu_adapt=0.02,reset_threshold=0.8
    def __init__(self, model,
                 alpha_init=0.003,
                 beta_init=1.5,
                 plasticity_decay=1,
                 lambda_adapt=0.02,
                 mu_adapt=0.01,
                 reset_threshold=0.7,
                 protect_percentile=90,
                 update_interval=1):
        """
        参数初始化
        :param model: 管理的神经网络模型
        :param alpha_init: 排名系数初始值 (控制Pareto前沿影响)
        :param beta_init: 拥挤度系数初始值 (控制解集多样性影响)
        :param plasticity_decay: 可塑性衰减因子
        +
        :param lambda_adapt: alpha的适应率 (控制探索到开发的转变)
        :param mu_adapt: beta的适应率 (控制多样性保护强度)
        :param reset_threshold: 重置概率阈值 (0.5-0.9)
        :param protect_percentile: 保护神经元的贡献度百分位
        :param update_interval: 更新间隔 (epoch数)
        """
        self.model = model
        self.alpha = alpha_init
        self.beta = beta_init
        self.plasticity_decay = plasticity_decay
        self.lambda_adapt = lambda_adapt
        self.mu_adapt = mu_adapt
        self.reset_threshold = reset_threshold
        self.protect_percentile = protect_percentile
        self.update_interval = update_interval

        # 神经元状态记录
        self.activation_records = defaultdict(list)
        self.gradient_records = defaultdict(list)
        self.hooks = []

        # 保护集和任务计数器
        self.protected_set = set()
        self.task_counter = 0
        self.epoch_count = 0

        # 注册钩子捕获激活和梯度
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向钩子捕获神经元激活和梯度"""
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

    def _forward_hook(self, module, inputs, outputs, name):
        """前向钩子：捕获并处理激活值"""
        if isinstance(module, nn.Conv2d):
            # 卷积层: 计算通道平均激活 (batch, C, H, W) -> (C,)
            channel_avg = outputs.detach().mean(dim=(0, 2, 3))
            self.activation_records[name].append(channel_avg)
        elif isinstance(module, nn.Linear):
            # 全连接层: 直接记录输出 (batch, D) -> (D,)
            self.activation_records[name].append(outputs.detach().mean(dim=0))

    def _backward_hook(self, module, grad_input, grad_output, name):
        """反向钩子：捕获并处理梯度"""
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

    def _calculate_metrics(self):
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
                contribution = (activations * gradients).mean(dim=0).abs().cpu().numpy()
            else:
                contribution = np.zeros_like(activation_strength)

            activation_metrics[name] = activation_strength
            contribution_metrics[name] = contribution

        return activation_metrics, contribution_metrics

    def _fast_non_dominated_sort(self, population):
        """
        NSGA-II快速非支配排序
        :param population: 神经元种群 [(layer_name, idx, M, C), ...]
        :return: 分层的Pareto前沿 [front1, front2, ...]
        """
        # 初始化支配关系数据结构
        S = {}  # 支配集合
        n = {}  # 被支配计数
        ranks = {}  # 前沿等级
        fronts = [[]]  # Pareto前沿

        # 第一遍遍历：初始化支配关系
        for p in population:
            p_key = (p[0], p[1])  # (layer_name, neuron_idx)
            S[p_key] = []
            n[p_key] = 0

            for q in population:
                q_key = (q[0], q[1])
                if p_key == q_key:
                    continue

                # 支配关系判断 (最小化M和C)
                # p支配q当且仅当: (M_p <= M_q 且 C_p <= C_q) 且至少一个严格不等式成立
                if (p[2] <= q[2] and p[3] <= q[3]) and (p[2] < q[2] or p[3] < q[3]):
                    S[p_key].append(q_key)
                elif (q[2] <= p[2] and q[3] <= p[3]) and (q[2] < p[2] or q[3] < p[3]):
                    n[p_key] += 1

            # 如果没有被任何其他解支配，则为第一前沿
            if n[p_key] == 0:
                ranks[p_key] = 0
                fronts[0].append(p)

        # 分层构建Pareto前沿
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                p_key = (p[0], p[1])
                for q_key in S[p_key]:
                    n[q_key] -= 1
                    if n[q_key] == 0:
                        # 找到q对应的神经元信息
                        q = next(item for item in population if (item[0], item[1]) == q_key)
                        ranks[q_key] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts

    def _calculate_crowding_distance(self, front):
        """
        修复的拥挤度距离计算 (解决全inf问题)
        :param front: 同一Pareto前沿的神经元列表
        :return: 每个神经元的拥挤度
        """
        size = len(front)
        if size == 0:
            return {}

        # 初始化距离字典 - 所有点初始化为0
        distances = {(neuron[0], neuron[1]): 0.0 for neuron in front}

        # 对每个目标函数进行处理
        for m in range(2):  # 目标索引: 0=M(活跃度), 1=C(贡献度)
            # 按当前目标值排序
            front.sort(key=lambda x: x[2 + m])  # 2+m 对应M或C的索引

            # 获取目标值的范围
            min_val = front[0][2 + m]
            max_val = front[-1][2 + m]

            # 处理所有值相同的情况
            if abs(max_val - min_val) < 1e-10:
                # 所有值相同，跳过此目标
                continue

            # 设置边界点的距离 (不再设为inf)
            # 对于只有1个或2个点的前沿，所有点都是边界点
            if size == 1:
                # 单个点，距离为0
                continue
            elif size == 2:
                # 两个点，距离相等
                for i in range(size):
                    key = (front[i][0], front[i][1])
                    distances[key] += 1.0  # 给予固定值
                continue

            # 对于3个及以上点的前沿
            # 边界点设置为最大可能距离 (而不是inf)
            key_first = (front[0][0], front[0][1])
            key_last = (front[-1][0], front[-1][1])
            distances[key_first] += 1.0
            distances[key_last] += 1.0

            # 计算中间点的拥挤度
            norm_factor = max_val - min_val
            for i in range(1, size - 1):
                # 计算当前点与前后的距离
                prev_val = front[i - 1][2 + m]
                next_val = front[i + 1][2 + m]
                distance_val = (next_val - prev_val) / norm_factor

                # 累加到总距离
                key = (front[i][0], front[i][1])
                distances[key] += distance_val

        return distances

    def _safe_exp(self, x):
        """
        安全的指数函数计算，防止溢出
        :param x: 输入值
        :return: 裁剪后的指数值
        """
        # 限制输入值范围，防止指数溢出
        clipped_x = np.clip(x, -50.0, 50.0)
        return math.exp(clipped_x)

    def _safe_sigmoid(self, z):
        """
        数值稳定的sigmoid函数计算
        :param z: 输入值
        :return: sigmoid概率值
        """
        if z > 0:
            return 1 / (1 + self._safe_exp(-z))
        else:
            return self._safe_exp(z) / (1 + self._safe_exp(z))

    def _moo_decision(self, activation_metrics, contribution_metrics):
        """
        基于多目标优化的神经元决策
        :return: (reset_candidates, protect_candidates)
        """
        reset_candidates = []
        protect_candidates = []
        neuron_population = []

        # 构建神经元种群 (层, 索引, M, C)
        for layer_name, M in activation_metrics.items():

            # M = (M - M.min()) / (M.max() - M.min() + (M.min()/100))

            C = contribution_metrics[layer_name]
            # C = (C - C.min()) / (C.max() - C.min() + (C.min()/100))

            num_neurons = len(M)

            for i in range(num_neurons):
                # 只考虑非保护神经元
                if (layer_name, i) not in self.protected_set:
                    neuron_population.append((layer_name, i, M[i], C[i]))

        # 如果没有神经元需要评估，返回空结果
        if not neuron_population:
            return reset_candidates, protect_candidates

        # 步骤1: 快速非支配排序
        fronts = self._fast_non_dominated_sort(neuron_population)
        # 步骤2: 计算拥挤度
        crowding_distances = {}
        rank_num = len(fronts)
        for front in fronts:
            if front:  # 跳过空前沿
                front_distances = self._calculate_crowding_distance(front)
                crowding_distances.update(front_distances)

        # 步骤3: 决策过程
        for neuron in neuron_population:
            layer_name, idx, M_val, C_val = neuron
            neuron_key = (layer_name, idx)

            # 获取Pareto等级和拥挤度
            rank = next((i for i, front in enumerate(fronts) if neuron in front), -1)
            # rank = rank/rank_num
            crowding = crowding_distances.get(neuron_key, 0.0)
            # 计算重置概率 (使用sigmoid函数)
            z = self.alpha * rank - self.beta * crowding
            reset_prob = self._safe_sigmoid(-z)  # 注意: 使用-z因为我们想要高z对应低概率
            # reset_prob = 1 / (1 + math.exp(self.alpha * rank - self.beta * crowding))
            # print(rank, crowding, reset_prob)

            # 重置决策: 高概率且非保护神经元
            if reset_prob > self.reset_threshold:
                reset_candidates.append((layer_name, idx))

        # 保护决策: 高贡献神经元
        for layer_name, C in contribution_metrics.items():
            if len(C) > 0:
                protect_thresh = np.percentile(C, self.protect_percentile)
                for i, c_val in enumerate(C):
                    if c_val > protect_thresh:
                        protect_candidates.append((layer_name, i))

        return reset_candidates, protect_candidates

    def _reset_neurons(self, reset_candidates):
        """重置选定的神经元"""
        for layer_name, neuron_idx in reset_candidates:
            module = dict(self.model.named_modules())[layer_name]

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

    def _apply_protection(self):
        """应用保护机制：将保护神经元的梯度置零"""
        for (layer_name, neuron_idx) in self.protected_set:
            module = dict(self.model.named_modules())[layer_name]

            if isinstance(module, nn.Conv2d):
                if module.weight.grad is not None:
                    module.weight.grad[neuron_idx] = 0
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.grad[neuron_idx] = 0

            elif isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    module.weight.grad[neuron_idx] = 0
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.grad[neuron_idx] = 0

    def _update_adaptation(self):
        """更新多目标优化参数"""
        # 指数衰减alpha (降低前沿等级影响)
        self.alpha = self.alpha * math.exp(-self.lambda_adapt * self.task_counter)
        # 渐进增加beta (提高拥挤度影响)
        self.beta = self.beta * (1 + self.mu_adapt * self.task_counter)

    def step(self):
        """执行神经元生命周期管理"""
        self.epoch_count += 1

        # 只在更新间隔执行
        if self.epoch_count % self.update_interval != 0:
            self.activation_records.clear()
            self.gradient_records.clear()
            return

        # 计算指标
        activation_metrics, contribution_metrics = self._calculate_metrics()

        # 多目标优化决策
        reset_candidates, protect_candidates = self._moo_decision(
            activation_metrics, contribution_metrics
        )

        # 添加新保护神经元
        for neuron_id in protect_candidates:
            self.protected_set.add(neuron_id)

        # 执行神经元重置
        self._reset_neurons(reset_candidates)

        # 应用梯度保护
        # self._apply_protection()

        # 更新多目标优化参数
        self._update_adaptation()

        # 更新任务计数器
        self.task_counter += 1

        # 重置记录
        self.activation_records.clear()
        self.gradient_records.clear()

        # 打印统计信息
        print(f"Epoch {self.epoch_count}: "
              f"Reset {len(reset_candidates)} neurons, "
              f"Protected {len(protect_candidates)} new neurons, "
              f"Total protected: {len(self.protected_set)}, "
              f"Alpha: {self.alpha:.4f}, Beta: {self.beta:.4f}")

    def visualize_fronts(self, activation_metrics, contribution_metrics, layer_name):
        """
        可视化指定层的Pareto前沿
        :param layer_name: 要可视化的层名称
        """
        if layer_name not in activation_metrics or layer_name not in contribution_metrics:
            print(f"Layer {layer_name} not found in metrics")
            return

        M = activation_metrics[layer_name]
        C = contribution_metrics[layer_name]
        num_neurons = len(M)

        # 构建神经元种群
        neuron_population = [
            (layer_name, i, M[i], C[i])
            for i in range(num_neurons)
            if (layer_name, i) not in self.protected_set
        ]

        if not neuron_population:
            print(f"No non-protected neurons in layer {layer_name}")
            return

        # 执行非支配排序
        fronts = self._fast_non_dominated_sort(neuron_population)

        # 可视化
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(fronts)))

        for front_idx, front in enumerate(fronts):
            if not front:
                continue

            M_vals = [neuron[2] for neuron in front]
            C_vals = [neuron[3] for neuron in front]

            plt.scatter(
                M_vals, C_vals,
                color=colors[front_idx],
                label=f'Front {front_idx}',
                s=50, alpha=0.7
            )

            # 标记重置决策
            for neuron in front:
                layer, idx, M_val, C_val = neuron
                neuron_key = (layer, idx)
                rank = front_idx
                crowding = self._calculate_crowding_distance(front).get(neuron_key, 0)

                # 计算重置概率
                reset_prob = 1 / (1 + math.exp(self.alpha * rank - self.beta * crowding))

                if reset_prob > self.reset_threshold:
                    plt.scatter(M_val, C_val, s=150, edgecolors='red', facecolors='none', linewidth=2)

        # 添加标签和标题
        plt.xlabel('Activation Magnitude (M)', fontsize=12)
        plt.ylabel('Contribution (C)', fontsize=12)
        plt.title(f'Pareto Fronts in {layer_name}\n(Red circles = reset candidates)', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # 保存和显示
        plt.savefig(f'pareto_fronts_{layer_name}.png', dpi=300)
        plt.show()

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ======================= 使用示例 ======================= #
if __name__ == "__main__":
    # 1. 创建模型和优化器
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32 * 8 * 8, num_classes)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            return self.fc(x)


    model = SimpleCNN(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 2. 初始化MOO生命周期管理器
    moo_manager = MOO_NeuronLifecycleManager(
        model,
        alpha_init=3.0,
        beta_init=0.5,
        plasticity_decay=0.97,
        lambda_adapt=0.1,
        mu_adapt=0.05,
        reset_threshold=0.7,
        protect_percentile=90,
        update_interval=3
    )

    # 3. 模拟训练循环
    # 注意: 此处使用随机数据仅用于演示
    for epoch in range(10):
        print(f"\n=== Epoch {epoch + 1}/10 ===")

        # 模拟数据批次
        for batch_idx in range(5):  # 5 batches per epoch
            # 模拟输入数据 (batch_size=4)
            inputs = torch.randn(4, 3, 32, 32)
            labels = torch.randint(0, 10, (4,))

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 应用神经元生命周期管理
            moo_manager.step()

            # 优化器更新
            optimizer.step()

            print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

    # 4. 可视化某一层的Pareto前沿
    # 首先需要重新计算指标
    activation_metrics, contribution_metrics = moo_manager._calculate_metrics()
    moo_manager.visualize_fronts(activation_metrics, contribution_metrics, 'conv1')

    # 5. 清理钩子
    moo_manager.remove_hooks()