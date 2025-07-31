import numpy as np
from matplotlib import pyplot as plt


def plot_line(classes, accuracy, exp, num):
    # 绘制折线图
    # for i, acc in enumerate(accuracy):
    c0 = "#5470c6"
    c1 = "#ee6666"
    y1 = accuracy[0]
    y2 = accuracy[1]
    exp = exp.replace("backup_new/", "")
    plt.figure(figsize=(4, 3), dpi=500)
    plt.plot(classes, y1, label=exp, marker="8", ms=5, mfc="white", color=c0)  # 添加数据点的标记
    plt.plot(classes, y2, label=exp+"_ACIE", marker="8", mfc="white", ms=5, color=c1)  # 添加数据点的标记
    plt.fill_between(classes, y1=y1, y2=y2, where=(np.array(y1)>np.array(y2)), color=c0,  alpha=0.3, interpolate=True)
    plt.fill_between(classes, y1=y1, y2=y2, where=(np.array(y2)>np.array(y1)), color=c1,  alpha=0.3, interpolate=True)
    # 设置图表标题和坐标轴标签
    if num == 0:
        plt.title(f'Combined with {exp}')  # 图表标题
        plt.ylabel('Accuracy on Learned Classes (%)')  # y轴标签
    else:
        # plt.title('New classes accuracy curve')  # 图表标题
        plt.ylabel('Accuracy on Old Classes (%)')  # y轴标签

    plt.xlabel('Class Number')  # x轴标签

    # 添加图例（右下角）
    plt.legend(loc='upper right')  # loc 参数指定图例的位置
    # plt.ylim([0, 100])
    # 添加网格（可选）
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    # 显示图表
    plt.show()

def show_result(num):
    y_list = []
    # stages = [i for i in range(10, 110, 10)]
    stages = [i for i in range(5, 105, 5)]
    # exps = ['ewc', 'finetune', 'lwf', 'der', 'wa', 'icarl',replay]
    exps = ['backup_new/lwf']
    # exps = ['ewc']
    for exp in exps:
        with open(f"results/{exp}.txt", 'r', encoding='utf-8') as f:
            # lines1 top5_accuracy_history | lines2 top1_accuracy_history | lines3 fitness_history
            line = f.readlines()[num]
            data_list = line.strip().replace(",", "").split()
            data_list = list(map(float, data_list))
            y_list.append(data_list)
        with open(f"results/{exp}_nief.txt", 'r', encoding='utf-8') as f:
            # lines1 top5_accuracy_history | lines2 top1_accuracy_history | lines3 fitness_history
            line = f.readlines()[num]
            data_list = line.strip().replace(",", "").split()
            data_list = list(map(float, data_list))
            y_list.append(data_list)
    plot_line(stages, y_list, exps[0], num)

if __name__ == '__main__':
    # show_result(0) # 绘制已学过的类准确度曲线
    show_result(1) # 绘制新类准确度曲线
    # show_result(2) # 绘制旧类准确度曲线
