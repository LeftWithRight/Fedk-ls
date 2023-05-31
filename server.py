import datetime
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import random
from Models import Mnist_2NN, Mnist_CNN
from client import *
from client import Cluster
from model.WideResNet import WideResNet
from getData import *
from Block import Block
from reposion import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
# 客户端的数量
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
# 随机挑选的客户端的数量
parser.add_argument('-cf', '--cfraction', type=float, default=1,
                    help='C fraction, 0 means 1 client, 1 means total clients')
# 训练次数(客户端更新次数)
parser.add_argument('-E', '--epoch', type=int, default=2, help='local train epoch')
# batchsize大小
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
# 模型名称
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
# 学习率
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-dataset', "--dataset", type=str, default="mnist", help="需要训练的数据集")
# 模型验证频率（通信频率）
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=50, help='global model save frequency(of communication)')
# n um_comm 表示通信次数，此处设置为1k
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='../checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')
parser.add_argument('-threshold', '--thresholdValue', type=float, default=0.8, help='set the thresValue in repoison')
parser.add_argument('-dp', '--dropout', type=float, default=0.1, help='set the dropout level in net')
parser.add_argument('-op', '--opti', type=str, default='SGD', help='set the opti in net')
parser.add_argument('-poipro', '--poisonprob', type=float, default=0.8, help='poison data')
parser.add_argument('-revprob', '--reverseprob', type=float, default=0.4, help='set the reverprob')
parser.add_argument('-repoi', '--repoisonalgorithms', type=str, default='trimmed_mean', help='set the repoison algorithms')






def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__
    #全局标签
    globalP = getGlobalp()
    print(globalP)




    # -----------------------文件保存-----------------------#
    # 创建结果文件夹
    # test_mkdir("./result")
    # path = os.getcwd()
    # 结果存放test_accuracy中
    test_txt = open("test_accuracy.txt", mode="a")
    # global_parameters_txt = open("global_parameters.txt",mode="a",encoding="utf-8")
    # ----------------------------------------------------#
    # 创建最后的结果
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    # 初始化模型
    # mnist_2nn
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN(args['dropout'])
    # mnist_cnn
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    # ResNet网络
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    ## 如果有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    # 将Tenor 张量 放在 GPU上
    net = net.to(dev)

    '''
        回头直接放在模型内部
    '''
    # 定义损失函数
    loss_func = F.cross_entropy
    # 优化算法的，随机梯度下降法
    # 使用Adam下降法
    if args['opti'] == 'Adam':
        opti = optim.Adam(net.parameters(), lr=args['learning_rate'])
    elif args['opti'] == 'SGD':
        opti = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=0.9)

    ## 创建Clients群
    '''
        创建Clients群100个

        得到Mnist数据

        一共有60000个样本
        100个客户端
        IID：
            我们首先将数据集打乱，然后为每个Client分配600个样本。
        Non-IID：
            我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
            然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
            注： 我觉得着并不是真正意义上的Non—IID
    '''
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    clientList = list(myClients.clients_set.values())
    testDataLoader = myClients.test_data_loader
    globalP = getGlobalp()

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    # 每次随机选取10个Clients
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 得到全局的参数
    global_parameters = {}
    # net.state_dict()  # 获取模型参数以共享

    # 得到每一层中全连接层中的名称fc1.weight
    # 以及权重weights(tenor)
    # 得到网络每一层上
    for key, var in net.state_dict().items():
        # print("key:"+str(key)+",var:"+str(var))
        print("张量的维度:" + str(var.shape))
        print("张量的Size" + str(var.size()))
        global_parameters[key] = var.clone()

    k = 10
    dn = 100
    ln = 10
    LgdArray = []
    clustergroupDict = {}
    # 生成初始分簇
    print("准备k个初始分簇")
    for kclient in range(k):
        clustergroupDict['cluster{}'.format(kclient)] = Cluster()
    clustergroup = list(clustergroupDict.values())  # 分簇列表

    # 遍历客户端列表，计算lgd，降序；生成两个列表，一个是初始中心，一个是待分簇的列表
    for client in clientList:
        lgdClientGlobal = calculateLgd(client, globalLabelP=globalP)
        LgdArray.append(lgdClientGlobal)
    sorted_indices = torch.argsort(torch.tensor(LgdArray), descending=True)  # 存储lgd的列表，降序排列，返回在初始列表的索引
    clusterCentrol = sorted_indices[0:k]  # 节点下表
    waittingclients = sorted_indices[k:]  # 等待加入的下表
    print("clustercentrl:", clusterCentrol)
    print("waitingclietns:", waittingclients)

    # 生成k个初始分簇，使用k个客户端
    for centralclient, cluster in zip(clusterCentrol, clustergroup):
        cluster.addClient(clientList[centralclient], globalP)
    print(len(clustergroup))
    # 'cluster{}'.format(i).addClient('client{}'.format(i))
    # 对剩余的客户端聚类
    print("对剩余节点进行聚类")
    for waittingclient in waittingclients:  # waittingclients存储的是下标
        tempLgs = []
        for cluster in clustergroup:
            lgsRatio = calculateClusterLgsBeforeAdd(cluster, clientList[waittingclient], globalP)
            # lgsRatio = calculateClusterLgsBeforeAdd(cluster, clientList[waittingclient], globalP):
            print("查看lgsRation:", lgsRatio)
            tempLgs.append(lgsRatio)

        sortedIndices = torch.argsort(torch.tensor(tempLgs), descending=True)
        clusterIndex = sortedIndices[0]  # 要加入的分簇索引
        print('client{}'.format(waittingclient), clusterIndex)
        clustergroup[clusterIndex].addClient(clientList[waittingclient], globalP)
        # 完成聚类
    for i in range(k):
        print("cluster{}的长度为".format(i), len(clustergroup[i].client))

    # 模拟中毒攻击，进行标签反转
    if args['poisonprob']:
        pro = args['poisonprob']
        poison_number = int(k * pro)
        print("模拟中毒攻击，对{}个分簇标签反转".format(poison_number))
        indexslist = list(range(k))  # 创建包含0到9的列表
        random_index_numbers = random.sample(indexslist, poison_number)  # 从列表中随机选择3个不重复的数
        print(random_index_numbers)
        for poisonIndex in random_index_numbers:
            poison_cluster = clustergroup[poisonIndex]
            for poisonclient in poison_cluster.client:
                poisonclient.reverse_labels(args['reverseprob'])
        print("完成模拟标签反转攻击")

    blockchain = [Block.create_genesis_block(global_parameters)]
    # 通讯次数一共
    for globalEpoch in range(args['num_comm']):
        print("communicate round {}".format(globalEpoch + 1))
        sum_parameters = None
        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        # 返回每个Client更新后的参数
        '''
            import time
            import tqdm
            # 方法1
            # tqdm(list)方法可以传入任意list，如数组
            for i in tqdm.tqdm(range(100)):
               time.sleep(0.5)
               pass
            # 或 string的数组
            for char in tqdm.tqdm(['a','n','c','d']):
               time.sleep(0.5)
               pass
        '''


        #开始训练
        client_parameters = None
        for index, cluster in tqdm(enumerate(clustergroup), total=len(clustergroup)):
            local_parameters = global_parameters
            for clientInCluster in cluster.client:
                client_parameters = clientInCluster.localUpdate(args['epoch'], args['batchsize'], net, loss_func, opti, local_parameters)
                local_parameters = client_parameters
            # 上传分簇模型参数到区块链
            blockchainLength = len(blockchain)
            clustername = 'cluster{}'.format(index)
            accuracy = getAccuracy(net, local_parameters, testDataLoader)
            blockchain.append(Block(blockchain[blockchainLength - 1].hash, clustername + "创建一个新的区块并上传参数", local_parameters, datetime.datetime.now(), accuracy))
            print(clustername + "创建一个新的区块并上传参数")

        # 全局模型聚合
        if args['repoisonalgorithms'] == 'trainBehavior':
            global_parameters = trainBehavior(blockchain, args['thresholdValue'])
            print("使用训练行为模型验证，筛选出诚实节点")
        elif args['repoisonalgorithms'] == 'krum':
            arguments = Args(k, 0, 1)
            global_parameters = krum(blockchain, arguments)
            print("使用krum安全聚合算法完成聚合")
        elif args['repoisonalgorithms'] == 'trimmed_mean':
            arguments = Args(k, 2, 0.8)
            global_parameters = trimmed_mean(blockchain, arguments)
            # global_parameters = trimmed_mean(blockchain)
            print("使用trimmed_mean安全聚合算法完成聚合")

        #  加载Server在最后得到的模型参数
        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        loss_num = 0
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            loss1 = loss_func(preds, label)
            loss_num += loss1.item()
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print("\n" + 'accuracy: {}'.format(sum_accu / num))
        loss_value = loss_num / len(testDataLoader)
        print("loss:" + str(loss_value) + "\n")
        test_txt.write("\ncommunicate round " + str(globalEpoch + 1) + "  ")
        test_txt.write('accuracy: ' + str(float(sum_accu / num)) + "  ")
        test_txt.write("loss_value: " + str(float(loss_value)))
        # test_txt.close()

        if (globalEpoch + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

    test_txt.close()