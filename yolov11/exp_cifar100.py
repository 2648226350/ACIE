import warnings
from os.path import exists

import numpy as np

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import matplotlib.pyplot as plt

# data_cfg = r"imagenet100"
# data_cfg = r"cifar100"
data_cfg = r"C:/Users/HNU123/Downloads/ImageNet100"
# data_cfg = r"datasets/imagenet_mini"
base_model_plt = r"ultralytics/cfg/models/11/yolo11n-cls-plt.yaml"
# base_model_plt = r"runs/train/base_plt/weights/best.pt"
exp_model_plt = r"runs/train/exp_plt/weights/last.pt"
# base_model = "yolo11n-cls.pt"
base_model = "ultralytics/cfg/models/11/yolo11n-cls.yaml"
exp_model = r"runs/train/exp/weights/last.pt"

CLASS_ADD_NUM = 10
NUM_CLASSES = 100
EPOCH_PER_STAGE = 30


def train_model(model, current_classes, name, p=0.3, r=0.4):
    model.train(data=data_cfg,
                cache=False,
                imgsz=128,
                epochs=int(NUM_CLASSES / CLASS_ADD_NUM * EPOCH_PER_STAGE),
                # epochs=EPOCH_PER_STAGE,
                # fraction=0.1,
                weight_decay=0.0005 if "L2" in name else 0,
                used_classes= current_classes[:CLASS_ADD_NUM],  # 训练过程被实际加载到dataloader中的数据
                base_classes = current_classes,
                batch=512,
                close_mosaic=0,
                workers=2,
                device='cuda',
                optimizer='SGD',
                resume=False,  # 续训的话这里  填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name=name,
                type='plasticity',
                exist_ok=True,
                epoch_per_stage=EPOCH_PER_STAGE,
                class_add_num=CLASS_ADD_NUM,
                protect_threshold=p,
                reset_threshold=r
                )

    # metrics = model.val(
    #     data=data_cfg,  # 使用与训练时相同的数据集配置
    #     imgsz=32,  # 使用与训练时相同的图像尺寸
    #     batch=100,  # 可以根据你的硬件调整批次大小
    #     device='0',  # 使用GPU进行验证
    #     split='test',  # 指定验证集，通常为 'val' 或 'test'
    #     used_classes=current_classes,
    #     exist_ok=True
    # )
    # return metrics


def plot_line(classes, accuracy):
    # 绘制折线图
    plt.plot(classes, accuracy[::2], label='YOLO-PLT')  # 添加数据点的标记
    plt.plot(classes, accuracy[1::2], label='YOLO')  # 添加数据点的标记

    # 设置图表标题和坐标轴标签
    plt.title('Accuracy changes with increasing classes')  # 图表标题
    plt.xlabel('Classes')  # x轴标签
    plt.ylabel('Accuracy')  # y轴标签

    # 添加图例（右下角）
    plt.legend(loc='lower right')  # loc 参数指定图例的位置
    plt.ylim([0, 100])
    # 添加网格（可选）
    # plt.grid(True)
    # 显示图表
    plt.show()


def main():
    # 记录每阶段的准确度
    top5_accuracy_history = []
    fitness_history = []
    top1_accuracy_history = []
    classes_history = []
    # 划分类别阶段
    # stages = [i for i in range(0, NUM_CLASSES, CLASS_ADD_NUM)]
    # all_classes = np.random.permutation(NUM_CLASSES).tolist()
    all_classes = list(range(NUM_CLASSES))
    # model = YOLO(base_model)
    # train_model(model, all_classes, "exp")
    # model = YOLO(base_model)
    # train_model(model, all_classes, "exp_L2")
    pls = [0.1,0.3,0.5,0.7,0.9]
    rls = [0.1,0.3,0.5,0.7,0.9]
    for p in pls:
        for r in rls:
            model = YOLO(base_model)
            train_model(model, all_classes, f"exp_mng_p_{p}_r_{r}", p, r)


'''
    for stage_idx, start_idx in enumerate(stages):
        current_classes = all_classes[start_idx:start_idx+CLASS_ADD_NUM]
        classes_history.append(current_classes)

        print(f"\n=== 阶段 {stage_idx + 1}: 训练类别 {current_classes} ===")
        if stage_idx == 0:
            model_plt = YOLO(base_model_plt)
            model = YOLO(base_model)
        else:
            model_plt = YOLO(exp_model_plt)
            model = YOLO(exp_model)

        metrics_plt = train_model(model_plt, current_classes, "exp_plt")
        metrics = train_model(model, current_classes, "exp")
        # 偶数存带有plastic的结果，奇数存普通结果
        top5_accuracy_history.append(metrics_plt.results_dict['metrics/accuracy_top5']*100)
        top1_accuracy_history.append(metrics_plt.results_dict['metrics/accuracy_top1'] * 100)
        fitness_history.append(metrics_plt.results_dict['fitness']*100)

        top5_accuracy_history.append(metrics.results_dict['metrics/accuracy_top5']*100)
        top1_accuracy_history.append(metrics.results_dict['metrics/accuracy_top1'] * 100)
        fitness_history.append(metrics.results_dict['fitness']*100)

    # 将结果写入到txt文件中
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(' '.join(map(str, top5_accuracy_history)) + '\n')  # 写入第一行并换行
        file.write(' '.join(map(str, top1_accuracy_history)) + '\n')  # 写入第一行并换行
        file.write(' '.join(map(str, fitness_history)) + '\n')  # 写入第一行并换行

    plot_line(stages, top5_accuracy_history)
    plot_line(stages, top1_accuracy_history)
    plot_line(stages, fitness_history)
'''

if __name__ == '__main__':
    main()
    # plot_line([0,1,2,3,4], [80,83,75,81,89,90,99,94,89,72])
