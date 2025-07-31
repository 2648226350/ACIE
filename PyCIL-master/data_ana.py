import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 加载 CIFAR - 100 数据集
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

classes = trainset.classes

def random_sample():
    # 获取一批数据
    images, labels = next(iter(trainloader))

    # 绘制随机样本图像
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    axes = axes.flatten()
    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def spec_sample():
    # 假设要展示类别 'apple' 的图像
    target_class = 'apple'
    target_index = classes.index(target_class)

    # 筛选特定类别的图像
    target_images = []
    target_labels = []
    for image, label in trainset:
        if label == target_index:
            target_images.append(image)
            target_labels.append(label)
            if len(target_images) >= 16:  # 限制展示数量
                break

    # 绘制特定类别图像
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))
    axes = axes.flatten()
    for i, (image, label) in enumerate(zip(target_images, target_labels)):
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axes[i].set_title(classes[label])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def class_num():
    from collections import Counter

    # 统计类别数量
    labels = [label for _, label in trainset]
    class_counts = Counter(labels)

    # 绘制柱状图
    plt.figure(figsize=(15, 8))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class Index')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

    # 绘制饼图
    plt.figure(figsize=(10, 10))
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title('Class Proportion')
    plt.show()

def img_feature():
    # 选择一张图像
    image, _ = trainset[0]
    image = image.numpy()

    # 绘制直方图
    plt.figure(figsize=(10, 5))
    plt.hist(image.flatten(), bins=256, range=[0, 1])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Pixel Value Histogram')
    plt.show()

    # 计算灰度均值
    mean_values = []
    for image, _ in trainset:
        mean_values.append(image.mean())

    # 绘制箱线图和直方图
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].boxplot(mean_values)
    axs[0].set_title('Mean Value Boxplot')
    axs[1].hist(mean_values, bins=50)
    axs[1].set_title('Mean Value Histogram')
    plt.show()

def matrices():
    import numpy as np
    import seaborn as sns

    # 创建一个示例相似性矩阵（实际应用中需要根据图像特征计算）
    num_classes = 50
    similarity_matrix = np.random.rand(num_classes, num_classes)  # 这里使用随机矩阵作为示例

    # 绘制热力图
    plt.figure(figsize=(15, 10))
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis')
    plt.title('Class Similarity Matrix')
    plt.show()

if __name__ == '__main__':
    # img_feature()
    random_sample()
    # matrices()