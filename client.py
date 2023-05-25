import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
from torchvision import transforms

class client(object):
    def __init__(self, trainDataSet, labelTensor, labelP,labelNumber, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.labelTensor = labelTensor
        self.p = labelP
        self.labelNumber = labelNumber

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # 加载当前通信中最新全局参数
        # 传入网络模型，并加载global_parameters参数的
        Net.load_state_dict(global_parameters, strict=True)
        # 载入Client自有数据集
        # 加载本地数据
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        # 设置迭代次数
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
                loss = lossFun(preds, label)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        return Net.state_dict()

    # 标签反转
    def reverse_labels(self, reverseprob):
        images = self.train_ds.tensors[0]
        labels = self.train_ds.tensors[1]
        reversed_labels = torch.where(torch.rand(labels.size()) < reverseprob, labels.max() - labels, labels)
        self.train_ds = TensorDataset(images, reversed_labels)

    def local_val(self):
        pass


class ClientsGroup(object):
    '''
        param: dataSetName 数据集的名称
        param: isIID 是否是IID
        param: numOfClients 客户端的数量
        param: dev 设备(GPU)
        param: clients_set 客户端

    '''

    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):

        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        # 加载测试数据
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        '''
            然后将其划分为200组大小为300的数据切片,然后分给每个Client两个切片
        '''

        if self.is_iid:
            # 60000 /100 = 600/2 = 300
            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            # print("shard_size:"+str(shard_size))

            # np.random.permutation 将序列进行随机排序
            # np.random.permutation(60000//300=200)
            shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
            # 一共200个
            print("*" * 100)
            print(shards_id)
            print(shards_id.shape)
            print("*" * 100)
            for i in range(self.num_of_clients):

                ## shards_id1
                ## shards_id2
                ## 是所有被分得的两块数据切片
                # 0 2 4 6...... 偶数
                shards_id1 = shards_id[i * 2]
                # 0+1 = 1 2+1 = 3 .... 奇数
                shards_id2 = shards_id[i * 2 + 1]
                #
                # 例如shard_id1 = 10
                # 10* 300 : 10*300+300
                # 将数据以及的标签分配给该客户端
                data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

                #
                # np.vstack 是按照垂直方向堆叠
                # np.hstack: 按水平方向（列顺序）堆叠数组构成一个新的数组
                '''
                    In[4]:
                    a = np.array([[1,2,3]])
                    a.shape
                    # (1, 3)
    
                    In [5]:
                    b = np.array([[4,5,6]])
                    b.shape             
                    # (1, 3)
    
                    In [6]:
                    c = np.vstack((a,b)) # 将两个（1,3）形状的数组按垂直方向叠加
                    print(c)
                    c.shape # 输出形状为（2,3）
                    [[1 2 3]
                     [4 5 6]]
                    # (2, 3)
    
                '''

                local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_label = np.argmax(local_label, axis=1)

                # 获取标签向量
                labelTensor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                labelLength = len(local_label)
                for labelindex in range(labelLength):
                    labelValue = local_label[labelindex]
                    labelTensor[labelValue] += 1
                labelTensor = torch.tensor(labelTensor)  # 张量化
                label_p = labelTensor / labelLength

                # 创建一个客户端
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), labelTensor, label_p,
                                 labelLength, self.dev)
                # 为每一个clients 设置一个名字
                # client10
                self.clients_set['client{}'.format(i)] = someone
        else:
            shard_image_samelabel = {}
            shard_label_samelabel = {}
            shard_image_diff = {}
            shard_label_diff = {}
            total_image_diff = []
            total_label_diff = []
            shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
            for i_diff_label in range(10):
                for i_same_label in range(10):
                    shard_image_samelabel['image.{}.{}'.format(i_diff_label, i_same_label)] = train_data[i_diff_label*6000+i_same_label*shard_size: i_diff_label*6000+(i_same_label+1)*shard_size]
                    shard_label_samelabel['label.{}.{}'.format(i_diff_label, i_same_label)] = train_label[i_diff_label*6000+i_same_label*shard_size: i_diff_label*6000+(i_same_label+1)*shard_size]
                if len(total_image_diff) == 0:
                    total_image_diff = train_data[i_diff_label * 6000 + 3000:(i_diff_label + 1) * 6000]
                else:
                    total_image_diff = np.vstack(
                        (total_image_diff, train_data[i_diff_label * 6000 + 3000:(i_diff_label + 1) * 6000]))

                if len(total_label_diff) == 0:
                    total_label_diff = train_label[i_diff_label * 6000 + 3000:(i_diff_label + 1) * 6000]
                else:
                    total_label_diff = np.vstack((total_label_diff, train_label[i_diff_label * 6000 + 3000:(i_diff_label + 1) * 6000]))



            #将剩余一半的数据打乱
            order = np.arange(len(total_label_diff))
            np.random.shuffle(order)
            total_label_diff = total_label_diff[order]
            total_image_diff = total_image_diff[order]
            for i_diff_label in range(10):
                for j_diff_label in range(10):
                    shard_image_diff['image.{}.{}'.format(i_diff_label, j_diff_label)] = total_image_diff[i_diff_label*3000+j_diff_label*300:i_diff_label*3000+(j_diff_label+1)*shard_size]
                    shard_label_diff['label.{}.{}'.format(i_diff_label, j_diff_label)] = total_label_diff[i_diff_label*3000+j_diff_label*300:i_diff_label*3000+(j_diff_label+1)*shard_size]

            for i_index in range(10):
                for j_index in range(10):
                    shard_image_01 = shard_image_samelabel['image.{}.{}'.format(i_index, j_index)]
                    shard_image_02 = shard_image_diff['image.{}.{}'.format(i_index, j_index)]
                    shard_label_01 = shard_label_samelabel['label.{}.{}'.format(i_index, j_index)]
                    shard_label_02 = shard_label_diff['label.{}.{}'.format(i_index, j_index)]
                    local_data, local_label = np.vstack((shard_image_01, shard_image_02)), np.vstack((shard_label_01, shard_label_02))
                    local_label_argmax = np.argmax(local_label, axis=1)
                    # 获取标签向量
                    labelTensor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    labelLength = len(local_label_argmax)
                    for labelindex in range(labelLength):
                        labelValue = local_label_argmax[labelindex]
                        labelTensor[labelValue] += 1
                    labelTensor = torch.tensor(labelTensor)  # 张量化
                    label_p = labelTensor / labelLength

                    local_label = np.argmax(local_label, axis=1)
                    # 创建一个客户端
                    someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), labelTensor,
                                     label_p,
                                     labelLength, self.dev)
                    self.clients_set['client{}'.format(i_index*10+j_index)] = someone


class Cluster(object):
    def __init__(self):
        self.client = []
        self.labelTensor = torch.tensor([0] * 10)
        self.p = None
        self.labelNumber = 0
        self.lgs = None


    def addClient(self, client, globalP):
        self.client.append(client)
        tempLabelTensor = self.labelTensor + client.labelTensor
        labelNumber = self.labelNumber + client.labelNumber
        self.labelTensor = tempLabelTensor             #跟新标签分布
        self.labelNumber += labelNumber            #更新数量
        self.p = self.labelTensor / self.labelNumber    #更新标签分布率
        squared_sum = (self.p - globalP).pow(2).sum()
        lgd = torch.sqrt(squared_sum)
        self.lgs = 1 - lgd

    def updateLgs(self, client, globalP):
        labelTensor = [x + y for x, y in zip(self.labelTensor, client.labelTensor)]
        self.labelNumber = self.labelNumber + client.labelNumber
        self.p = torch.tensor(labelTensor) / client.labelNumber
        squared_sum = (self.p - globalP).pow(2).sum()
        lgd = torch.sqrt(squared_sum)
        self.lgs = 1 - lgd



# 计算设备与全局之间的lgd;用于k-lgs
def calculateLgd(client, globalLabelP):
    squared_sum = (client.p - globalLabelP).pow(2).sum()
    lgd = torch.sqrt(squared_sum)
    return lgd


#计算加入设备前后，分簇的lgs变化; 用于临时计算，不更新分簇信息
def calculateClusterLgsBeforeAdd(cluster, client, globalLabelP):
    beforelgs =cluster.lgs
    labelTensor = cluster.labelTensor + client.labelTensor
    labelNumber = cluster.labelNumber + client.labelNumber

    latestlabelP = labelTensor / labelNumber
    squared_sum = (latestlabelP - globalLabelP).pow(2).sum()
    lgd = torch.sqrt(squared_sum)
    lgs = 1 - lgd
    lgsRatio = lgs / cluster.lgs
    return lgsRatio


def getAccuracy(net, parameters, testDataLoader):
    net.load_state_dict(parameters, strict=True)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sum_accu = 0
    num = 0
    # 载入测试集
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
    return sum_accu / num