from getData import *
import os
import numpy as np
import torch

#全局标签
def getGlobalp():
    data_dir = r'.\data\MNIST'
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    train_labels = extract_labels(train_labels_path)
    # 统计每个标签的数量
    label_counts = [0] * 10  # 创建一个初始全零的10维列表
    train_labels = np.argmax(train_labels, axis=1)     # train_labels 为60000*10的二维列表
    for label in train_labels:
        label_counts[int(label)] += 1
    label_counts = torch.tensor(label_counts)
    globalP = label_counts / 60000.0
    return globalP


# if __name__ == "__main__":
#     # 打印每个标签对应的数量
#     for i, count in enumerate(label_counts):
#         print("标签 {}: 数量 {}".format(i, count))
#
#     # 存储在一个10维的列表中
#     label_counts_list = list(label_counts)
#     print(label_counts_list)