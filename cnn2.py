#coding: utf-8
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import time
import ipdb
import tarfile
import os.path
import cPickle
import six
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer import function
from load_data import load_data
from six.moves.urllib import request

def unpickle(f):
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def load_cifar10(datadir="cifar-10-batches-py"):
    # CIFAR-10 データセットがなければダウンロードする
    if os.path.exists(datadir) == False:
        print("Downloading cifar-10...")
        request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz","cifar10.tar.gz")
        tar = tarfile.open("cifar10.tar.gz")
        tar.extractall()
        tar.close()

    train_data = []
    train_target = []

    # 訓練データをロード
    for i in range(1, 6):
        d = unpickle("%s/data_batch_%d" % (datadir, i))
        train_data.extend(d["data"])
        train_target.extend(d["labels"])

    # テストデータをロード
    d = unpickle("%s/test_batch" % (datadir))
    test_data = d["data"]
    test_target = d["labels"]

    # データはfloat32、ラベルはint32のndarrayに変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int32)

    # 画像のピクセル値を0-1に正規化
    train_data /= 255.0
    test_data /= 255.0

    return train_data, test_data, train_target, test_target

# 畳込みニューラルネットワーク
class CNN(chainer.link.Chain):
    def __init__(self, n_out=10):
        # ネットワーク構成
        super(CNN, self).__init__(
            # 入力チャネル数3 出力チャネル数32 フィルタサイズ3
            conv1=L.Convolution2D(3, 32, 3, pad=1), # 縦32 * 横32 --conv--> 縦32 * 横32 --pooling--> 縦16 * 横16
            # 入力チャネル数32 出力チャネル数32 フィルタサイズ3
            conv2=L.Convolution2D(32, 32, 3, pad=1), # 縦16 * 横16 --conv--> 縦16 * 横16 --pooling--> 縦8 * 横8
            # 縦8 * 横8 * チャネル数32 = 2048
            fc1=L.Linear(2048, 1024),
            fc2=L.Linear(1024, n_out),
        )
    
    def forward(self, x_data, y_data, train=True):
        x, t = chainer.Variable(xp.array(x_data)), chainer.Variable(xp.array(y_data))
        h = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
        h = F.dropout(F.relu(model.fc1(h)), train=train)
        y = model.fc2(h)

        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return F.accuracy(y, t)


if __name__ == "__main__":
    # GPUを使う
    gpu_flag = 0
    if gpu_flag >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_flag >= 0 else np

    batchsize = 100 # 一度に学習するデータの数
    n_epoch = 20 # トレーニング回数

    # クラスの数
    n_out = 10 # cifar10なら10クラスなので10．

    dataset = 'cifar10' # 使うデータセットを指定

    if dataset == 'mine':
        # pathで指定したディレクトリのデータセットをロード
        print("load dataset")
        X, Y= load_data(path="./dataset",mode='c')
        X_train, X_test = X[0], X[1]
        y_train, y_test = Y[0], Y[1] 
        N = X_train.shape[0]
        N_test = X_test.shape[0]

    elif dataset == 'cifar10':
        # CIFAR-10データセットをロード
        print "load CIFAR-10 dataset"
        X_train, X_test, y_train, y_test = load_cifar10()
        N = y_train.size
        N_test = y_test.size

        # 画像を (nsample, channel, height, width) の4次元テンソルに変換
        X_train = X_train.reshape((len(X_train), 3, 32, 32))
        X_test = X_test.reshape((len(X_test), 3, 32, 32))


    model = CNN(n_out)
    # 学習済モデルをロードする場合は以下のコメントを外す
    #serializers.load_npz('cnn.model', model)
    
    if gpu_flag >= 0:
        cuda.get_device(gpu_flag).use()
        model.to_gpu()

    # オプティマイザーを設定 
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # 結果を書き込むファイルを作成
    fp1 = open("accuracy.txt", "w")
    fp2 = open("loss.txt", "w")

    fp1.write("epoch\ttest_accuracy\n")
    fp2.write("epoch\ttrain_loss\n")

    # epochで指定した回数だけ訓練ループ
    for epoch in range(1, n_epoch + 1):
        print "epoch: %d" % epoch

        perm = np.random.permutation(N)
        sum_loss = 0 # 誤差を入れる変数

        # 訓練データでモデルを学習
        for i in range(0, N, batchsize):
            # 一度に使うのは，batchsize分のデータだけ
            if dataset == 'mine':
                x_batch = xp.asarray(X_train[i:i + batchsize])
                y_batch = xp.asarray(y_train[i:i + batchsize])
            elif dataset == 'cifar10':
                x_batch = xp.asarray(X_train[perm[i:i + batchsize]])
                y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

            # オプティマイザーを初期化
            optimizer.zero_grads()
            # ネットワークを順伝播して誤差を求める
            loss = model.forward(x_batch, y_batch)
            # 誤差逆伝播
            loss.backward()
            # 重みを変更(ネットワークを学習)
            optimizer.update()
            sum_loss += float(loss.data) * len(y_batch)

        print "train mean loss: %f" % (sum_loss / N)
        fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
        fp2.flush()

        # テストデータで識別率を検証
        sum_accuracy = 0
        for i in range(0, N_test, batchsize):
            x_batch = xp.asarray(X_test[i:i + batchsize])
            y_batch = xp.asarray(y_test[i:i + batchsize])

            acc = model.forward(x_batch, y_batch, train=False)
            sum_accuracy += float(acc.data) * len(y_batch)

        print "test accuracy: %f" % (sum_accuracy / N_test)
        fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
        fp1.flush()


    fp1.close()
    fp2.close()

    # 学習したモデルを保存
    serializers.save_npz('cnn.model', model)
