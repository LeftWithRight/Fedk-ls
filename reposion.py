import torch
import numpy as np
from collections import defaultdict
import copy



class Args:
    def __init__(self, num_users, atk_num, frac):
        self.num_users = num_users
        self.atk_num = atk_num
        self.frac = frac

    def num_none_poison(self):
        print(str(self.num_users - self.atk_num))


def pearson_sim(param_diff1, param_diff2):
    similarity = 0
    for key in param_diff1.keys():
        diff1 = param_diff1[key]
        diff2 = param_diff2[key]
        similarity += torch.sum((diff1 - diff1.mean()) * (diff2 - diff2.mean())) / (
                torch.sqrt(torch.sum(torch.pow(diff1 - diff1.mean(), 2))) *
                torch.sqrt(torch.sum(torch.pow(diff2 - diff2.mean(), 2))) +
                1e-8
        )
    similarity /= len(param_diff1)
    # print('accuracy: {}'.format(similarity))
    return similarity


# def trainBehavior(blockchain, threshold):
#     length = len(blockchain)
#     max_acc = 0
#
#     # 获取准确率最好的索引
#     max_index = None
#     for i in range(length - 10, length):
#         if max_acc < blockchain[i].get_accu():
#             max_acc = blockchain[i].get_accu()
#             max_index = i
#     lst = []
#     thresholdVlaue = threshold * max_acc
#     print("\n" + 'thresholdVlaue: {}'.format(thresholdVlaue))
#     for i in range((length-10), length):
#         if blockchain[i].get_accu() > thresholdVlaue and pearson_sim(blockchain[i].params, blockchain[max_index].params) > threshold:
#             lst.append(i)
#     print("参与此次模型聚合的局部模型数量：" + str(len(lst)))
#     distances = defaultdict(dict)
#     non_malicious_count = len(lst)
#     num = 0
#     w = []
#     for i in lst:
#         w.append(blockchain[i].get_para())
#     num_models = len(w)
#     for k in w[0].keys():
#         if num == 0:
#             for i in range(len(w)):
#                 for j in range(i):
#                     distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
#             num = 1
#         else:
#             for i in range(len(w)):
#                 for j in range(i):
#                     distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
#                     distances[i][j] += distances[j][i]
#     minimal_error = 1e20
#     minimal_error_index = 0
#     for user in distances.keys():
#         errors = sorted(distances[user].values())
#         current_error = sum(errors[:non_malicious_count])
#         if current_error < minimal_error:
#             minimal_error = current_error
#             minimal_error_index = user
#     return w[minimal_error_index]

def trainBehavior(blockchain, threshold):
    length = len(blockchain)
    max_acc = 0

    # 获取准确率最好的索引
    max_index = None
    for i in range(length - 10, length):
        if max_acc < blockchain[i].get_accu():
            max_acc = blockchain[i].get_accu()
            max_index = i

    lst = []
    thresholdVlaue = threshold * max_acc
    print("\n" + 'thresholdVlaue: {}'.format(thresholdVlaue))
    for i in range((length-10), length):
        # print("\n" + 'simlarity {}'.format(pearson_similarity(blockchain[i].bp, blockchain[max_index].bp)))
        # similarity = F.cosine_similarity(blockchain[i].bp, blockchain[max_index].bp, dim=0)
        # print(pearson_sim(blockchain[i].bp, blockchain[max_index].bp))
        if blockchain[i].get_accu() > thresholdVlaue and pearson_sim(blockchain[i].params, blockchain[max_index].params) > threshold:
            lst.append(i)
    print("参与此次模型聚合的局部模型数量：" + str(len(lst)), "区块链中的下标", lst)
    w = []
    for i in lst:
        w.append(blockchain[i].params)
    num_models = len(lst)
    num_non_malicious = num_models
    distances = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(i):
            dist = 0
            for param_name, param_value in w[i].items():
                t = param_value.cpu().numpy()
                t1 = w[j][param_name].cpu().numpy()
                dist += np.linalg.norm(t - t1)
                # dist += torch.norm(param_value - w[j][param_name]).item()
            distances[i, j] = distances[j, i] = dist

    sorted_distances = np.sort(distances, axis=1)
    selected_distances = sorted_distances[:, :num_non_malicious]

    # 计算选择的距离之和
    errors = np.sum(selected_distances, axis=1)

    # 找到最小错误值对应的模型索引
    krum_index = np.argmin(errors)

    return w[krum_index]


def krum(blockchain, args):  # w中存储的是所有分簇的模型参数 args存储标签反转数据信息
    print("使用Krum算法进行模型聚合")
    length = len(blockchain)
    w = []
    for i in range(length-10, length):
        w.append(blockchain[i].get_para())
    num_models = len(w)
    num_non_malicious = int((args.num_users - args.atk_num))
    distances = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(i):
            dist = 0
            for param_name, param_value in w[i].items():
                dist += np.linalg.norm(param_value.cpu().numpy() - w[j][param_name].cpu().numpy())
            distances[i, j] = distances[j, i] = dist

    sorted_distances = np.sort(distances, axis=1)
    selected_distances = sorted_distances[:, :num_non_malicious]

    # 计算选择的距离之和
    errors = np.sum(selected_distances, axis=1)

    # 找到最小错误值对应的模型索引
    krum_index = np.argmin(errors)

    return w[krum_index]


def medium(blockchain, args):
    length = len(blockchain)
    w = []
    for i in range(length - 10, length):
        w.append(blockchain[i].get_para())
    distances = defaultdict(dict)
    non_malicious_count = int((args.num_users - args.atk_num) * args.frac)
    num = 0
    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())   # 计算相似度
                    distances[i][j] += distances[j][i]
    errorlst = []
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        errorlst.append(current_error)
    newerrorlst = sorted(errorlst)
    return newerrorlst[4]  # 返回中位数


def trimmed_mean(blockchain, args):
    length = len(blockchain)
    w = []
    for i in range(length - 10, length):
        w.append(blockchain[i].get_para())
    number_to_consider = int((args.num_users - args.atk_num) * args.frac) - 1
    print(number_to_consider)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = []
        for i in range(len(w)):
            tmp.append(w[i][k].numpy()) # get the weight of k-layer which in each client
        tmp = np.array(tmp)
        med = np.median(tmp, axis=0)
        new_tmp = []
        for i in range(len(tmp)):             # cal each client weights - median
            new_tmp.append(tmp[i]-med)
        new_tmp = np.array(new_tmp)
        good_vals = np.argsort(np.abs(new_tmp), axis=0)[:number_to_consider]
        good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
        k_weight = np.array(np.mean(good_vals) + med)
        w_avg[k] = torch.from_numpy(k_weight)
    return w_avg


# def trimmed_mean(blockchain):
#     length = len(blockchain)
#     param_list = []
#     for i in range(length - 10, length):
#         param_list.append(blockchain[i].get_para())
#     num_params = len(param_list)
#     num_to_consider = int(num_params * 0.9)  # Consider 90% of the parameters
#
#     aggregated_params = copy.deepcopy(param_list[0])  # Initialize aggregated params as a deep copy of the first parameter dict
#
#     # Calculate trimmed mean for each parameter in the list
#     for key in aggregated_params.keys():
#         param_vals = [param[key] for param in param_list]
#         param_vals = [val.numpy() if isinstance(val, torch.Tensor) else val for val in
#                       param_vals]  # Convert tensors to numpy arrays
#         sorted_vals = np.sort(param_vals)
#         trimmed_vals = sorted_vals[:num_to_consider]  # Take the lowest num_to_consider values
#         trimmed_mean = np.mean(trimmed_vals)  # Calculate the trimmed mean
#         trimmed_mean = torch.tensor(trimmed_mean)  # Convert back to torch.Tensor
#
#
#         aggregated_params[key] = trimmed_mean  # Set the aggregated parameter value to the trimmed mean
#
#     return aggregated_params
