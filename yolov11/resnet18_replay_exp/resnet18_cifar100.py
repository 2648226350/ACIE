import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from NeuronLifecycleManager import NeuronLifecycleManager


# 实验参数
NUM_TASKS = 20  # 总共10个任务
CLASSES_PER_TASK = 5  # 每个任务5个类
INITIAL_CLASSES = 5  # 初始类别数
TOTAL_CLASSES = NUM_TASKS * CLASSES_PER_TASK  # 共50个类 (CIFAR100子集)
EPOCH_PER_STAGE = 200
WEIGHT_DECAY = 0.0005


# 模型架构 (简化的ResNet)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# 数据准备
def prepare_cifar100_data():
    print("preparing data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=False, transform=transform_train)

    full_test = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=transform_test)

    # 选择50个类作为子集（CIFAR100有100类）
    # selected_classes = np.random.choice(100, TOTAL_CLASSES, replace=False)
    selected_classes = range(TOTAL_CLASSES)
    class_mapping = {cls: i for i, cls in enumerate(selected_classes)}

    # # 筛选训练集
    # train_indices = [i for i, (_, label) in enumerate(full_train) if label in selected_classes]
    # train_subset = torch.utils.data.Subset(full_train, train_indices)

    # # 筛选测试集
    # test_indices = [i for i, (_, label) in enumerate(full_test) if label in selected_classes]
    # test_subset = torch.utils.data.Subset(full_test, test_indices)

    # # 创建任务数据集
    # task_datasets = []
    # for task_id in range(NUM_TASKS):
    #     print(f"creating task-{task_id} dataset...")
    #     task_classes = selected_classes[task_id * CLASSES_PER_TASK:(task_id + 1) * CLASSES_PER_TASK]
    #     task_indices = [i for i, (_, label) in enumerate(train_subset)
    #                     if label in task_classes]
    #     task_datasets.append(torch.utils.data.Subset(train_subset, task_indices))

    return full_train, full_test, class_mapping


def get_task_dataset(task_id, full_train, full_test):
    print(f"preparing {task_id+1}-task dataset...")

    # 选择100个类作为子集（CIFAR100有100类）
    selected_classes = range(TOTAL_CLASSES)
    task_classes = selected_classes[:(task_id + 1) * CLASSES_PER_TASK]

    print("get data taskclasses", task_classes)
    # 筛选测试集
    test_indices = [i for i, (_, label) in enumerate(full_test) if label in task_classes]

    # 筛选训练集
    train_indices = [i for i, (_, label) in enumerate(full_train) if label in task_classes]

    return torch.utils.data.Subset(full_train, train_indices), torch.utils.data.Subset(full_test, test_indices)


# 训练函数
def train_model(model, train_loader, optimizer, criterion, device, manager=None, epoch=1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for idx, (inputs, labels) in enumerate(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    return total_loss / len(train_loader), 100. * correct / total


def model_eval(model, test_loader, device, learned_classes=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 如果只测试已学习的类别
            # if learned_classes is not None:
            #     mask = torch.tensor([label in learned_classes for label in labels.cpu()]).to(device)
            #     inputs = inputs[mask]
            #     labels = labels[mask]
            #     outputs = outputs[:, list(learned_classes)]
            #     if inputs.size(0) == 0:
            #         continue

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


# 实验主函数
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_train, full_test, class_mapping = prepare_cifar100_data()

    # 三种实验条件
    conditions = {
        # "baseline":{"incremental": False, "manager": False},
        # "Incremental": {"incremental": True, "manager": False},
        "Incremental+NLM": {"incremental": True, "manager": True}
    }

    results = {name: [] for name in conditions}
    models = {}

    for cond_name, config in conditions.items():
        print(f"\n=== Running {cond_name} condition ===")

        if not config["incremental"]:
            learned_classes = set()
            criterion = nn.CrossEntropyLoss()

            for task_id in range(NUM_TASKS):
                print(f"\n--- Task {task_id + 1}/{NUM_TASKS} ---")
                model = resnet18(num_classes=TOTAL_CLASSES).to(device)
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=WEIGHT_DECAY)
                # 当前任务的类别
                task_classes = set(range(
                    task_id * CLASSES_PER_TASK,
                    (task_id + 1) * CLASSES_PER_TASK
                ))
                # print("task_classes", task_classes)
                learned_classes |= task_classes

                task_train, task_test = get_task_dataset(task_id, full_train, full_test)

                # 准备当前任务数据
                # task_loader = torch.utils.data.DataLoader(
                #     task_datasets[task_id], batch_size=128, shuffle=True, num_workers=2)
                task_loader = torch.utils.data.DataLoader(
                    task_train, batch_size=256, shuffle=True, num_workers=8)
                test_loader = torch.utils.data.DataLoader(
                    task_test, batch_size=128, shuffle=True, num_workers=8)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

                for epoch in range(EPOCH_PER_STAGE):
                    train_loss, train_acc = train_model(model, task_loader, optimizer, criterion, device)
                    scheduler.step()

                    if (epoch + 1) % 10 == 0:
                        print(f"Epoch [{epoch + 1}/{EPOCH_PER_STAGE}] | Loss: {train_loss:.4f} | "
                              f"Train Acc: {train_acc:.2f}%")

                test_acc = model_eval(model, test_loader, device, learned_classes)
                results[cond_name].append(test_acc)
                print(f"After Task {task_id + 1} | Test Acc on Learned Classes: {test_acc:.2f}%")


                
            '''
            # Baseline: 一次性训练所有类别
            model = resnet18(num_classes=TOTAL_CLASSES).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=WEIGHT_DECAY)
            criterion = nn.CrossEntropyLoss()

            # 合并所有任务数据
            # full_train = torch.utils.data.ConcatDataset(task_datasets)
            train_loader = torch.utils.data.DataLoader(
                full_train, batch_size=256, shuffle=True, num_workers=2)
                
            test_loader = torch.utils.data.DataLoader(full_test, batch_size=256, shuffle=False)

            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

            for epoch in range(EPOCH_PER_STAGE):
                train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
                test_acc = model_eval(model, test_loader, device)
                scheduler.step()

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{EPOCH_PER_STAGE}] | Loss: {train_loss:.4f} | "
                          f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

            results[cond_name] = [test_acc] * NUM_TASKS
            models[cond_name] = deepcopy(model)
            '''

        else:
            # 增量学习场景
            model = resnet18(num_classes=TOTAL_CLASSES).to(device)
            
            criterion = nn.CrossEntropyLoss()

            # 初始化神经元生命周期管理器 (仅用于NLM条件)
            manager = None
            if config["manager"]:
                manager = NeuronLifecycleManager(
                    model = model,
                    plasticity_factor=0.5,
                    protection_factor=0.5,
                    reset_threshold=0.2,
                    protect_threshold=0.3,
                    plasticity_decay=1,
                    update_interval=1
                )


            learned_classes = set()

            for task_id in range(NUM_TASKS):
                print(f"\n--- Task {task_id + 1}/{NUM_TASKS} ---")
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=WEIGHT_DECAY)
                # 当前任务的类别
                task_classes = set(range(
                    task_id * CLASSES_PER_TASK,
                    (task_id + 1) * CLASSES_PER_TASK
                ))
                # print("task_classes", task_classes)
                learned_classes |= task_classes

                task_train, task_test = get_task_dataset(task_id, full_train, full_test)

                # 准备当前任务数据
                # task_loader = torch.utils.data.DataLoader(
                #     task_datasets[task_id], batch_size=128, shuffle=True, num_workers=2)
                task_loader = torch.utils.data.DataLoader(
                    task_train, batch_size=1024, shuffle=True, num_workers=12)
                test_loader = torch.utils.data.DataLoader(
                    task_test, batch_size=512, shuffle=True, num_workers=12)

                # 训练当前任务
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

                for epoch in range(EPOCH_PER_STAGE):
                    train_loss, train_acc = train_model(
                        model, task_loader, optimizer, criterion, device, manager, epoch)

                    scheduler.step()

                    if (epoch + 1) % 10 == 0:
                        print(f"Epoch [{epoch + 1}/{EPOCH_PER_STAGE}] | Loss: {train_loss:.4f} | "
                              f"Train Acc: {train_acc:.2f}%")

                # 测试所有已学类别
                test_acc = model_eval(model, test_loader, device, learned_classes)
                # 如果使用神经元生命周期管理
                if manager:
                    manager.step(len(test_loader))
                results[cond_name].append(test_acc)
                print(f"After Task {task_id + 1} | Test Acc on Learned Classes: {test_acc:.2f}%")

                # 保存模型快照
                models[f"{cond_name}_task{task_id + 1}"] = deepcopy(model)

    return results, models
    
def save_results(results):
    path = "/home/jczn2/yolov11/NIEF/resnet18_cifar100_imgsz64_epoch200_plastic_incre5/"
    for cond_name, acc_list in results.items():
        with open(path+f"{cond_name}.txt", "w", encoding="utf-8") as f:
            f.write(str(acc_list).strip("[]"))


def show_results():
    results = {}
    path = "resnet18_cifar100_imgsz64_epoch200_plastic_incre5/"
    files = ["Incremental","Incremental+NLM", "baseline"]
    for file in files:
        with open(path+file+".txt", 'r', encoding="utf-8") as f:
            results[file]=f.read().split(",")

    plt.figure(figsize=(12, 8))

    for cond_name, acc_list in results.items():
        acc_list = [float(x) for x in acc_list]
        print(cond_name, sum(acc_list)/len(acc_list))
        plt.plot(acc_list, 'o-', label=cond_name)
    
    plt.title('Incremental Learning Performance on CIFAR100', fontsize=14)
    plt.xlabel('Task Number', fontsize=12)
    plt.ylabel('Accuracy on Learned Classes (%)', fontsize=12)
    plt.xticks(range(NUM_TASKS), range(1, NUM_TASKS + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # 保存图表
    plt.savefig('incremental_learning_results.png', dpi=500)
    plt.show()

# 运行实验并可视化结果
if __name__ == "__main__":
    # results, models = run_experiment()
    # results = {"cod":[1,2,3,4]}
    # save_results(results)
    show_results()  
