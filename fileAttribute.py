import os
import gzip
import numpy as np

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

# 加载数据集
data_dir = r'.\data\MNIST'
# data_dir = r'./data/MNIST'
# python路径拼接os.path.join() 路径变为.\data\MNIST\train-images-idx3-ubyte.gz
train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
train_images = extract_images(train_images_path)
print("-"*5+"train_images"+"-"*5)
# 输出第一张图片

#print(train_images[0].reshape(28,28))
print(train_images.shape) # (60000, 28, 28, 1) 一共60000 张图片，每一张是28*28*1
print('-'*22+"\n")
train_labels = extract_labels(train_labels_path)
print("-" * 5 + "train_labels" + "-" * 5)
print(train_labels.shape) # (60000, 10)
print('-'*22+"\n")


