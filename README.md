# Fedk-ls

The implementation of federated learning PyTorch respectively.



### environment





##### PyTorch-version

1.Python 3.9.13

2.pytorch 1.3.1

both of them run on GPU

### prepare data sets

You are supposed to prepare the data set by yourself. MNIST can be downloaded on http://yann.lecun.com/exdb/mnist/. These data sets should be put into /data/MNIST when the download is finished.

### usage

Run the code

```asp
python server.py -nc 100 -cf 1 -E 5 -B 128  -mn mnist_2nn  -ncomm 50 -iid 0 -lr 0.001 -vf 1 -g 1 -dp 0.5 -op SGD -poipro 0.5 -revprob 0.5 -threshold 0.8 -repoi trainBehavior
```

which means there are 100 clients.     The data set are allocated in Non-IID way.     The epoch and batch size are set to 5 and 128.    The learning rate is 0.01, we validate the codes every 20 rounds during the training, training stops after 50 rounds.    There are three models to do experiments: mnist_2nn mnist_cnn and cifar_cnn, and we choose mnist_cnn in this command.    Notice the data set path when run the code of pytorch-version(you can take the source code out of the 'use_pytorch' folder).
