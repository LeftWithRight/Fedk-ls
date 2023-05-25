import datetime
import hashlib
import copy


class Block:
    def __init__(self, previous_block_hash, transaction, params, timestamp, accuracy):
        self.previous_block_hash = previous_block_hash
        self.transaction = transaction
        self.params = copy.deepcopy(params)
        self.timestamp = timestamp
        self.accuracy = accuracy
        self.hash = self.get_hash()


    # 创建创世区块

    @staticmethod
    def create_genesis_block(globalpara):             #静态方法不用传入self
        print('生成创世区块，即将进入训练')
        return Block('0', '0', globalpara, datetime.datetime.now(), 0)



        # 返回hash值

    def get_hash(self):
        header_bin = str(self.previous_block_hash) + str(self.transaction) + str(self.timestamp)
        out_hash = hashlib.sha256(header_bin.encode()).hexdigest()  # 加密
        return out_hash

    def get_para(self):
        return self.params

    def get_accu(self):
        return self.accuracy