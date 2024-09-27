## サンプルGANの実装には次のQiita(https://qiita.com/takuto512/items/c8fd8eb1cdc7b6689798)を参考にした.

import numpy as np
import pandas as pd
import os, time, re
import pickle, gzip, datetime

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Grid

'''Data Prep and Model Evaluation'''
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error

'''Algos'''
import lightgbm as lgb

'''TensorFlow and Keras'''
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import LeakyReLU, Reshape, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Input, Lambda
from tensorflow.keras.layers import Embedding, dot
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


sns.set("talk")

# 外部からの学習データセット読み込み.
# 学習データとテストデータに分割したデータ
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
# ピクセルの値を 0~1 の間に正規化
x_train= x_train / 255.0


## 以下, DCGANクラス実装
#DCGANのクラス.
class DCGAN(object):
  #初期化
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    #生成ネットワーク
    #100*1*1の行列をデータセットの画像と同じ1*28*28にする
    def generator(self, depth=256, dim=7, dropout=0.3, momentum=0.8, \
                  window=5, input_dim=100, output_depth=1):
        if self.G:
            return self.G
        self.G = Sequential()

        #100*1*1 → 256*7*7
        self.G.add(Dense(dim*dim*depth, input_dim=input_dim))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        #256*7*7 → 128*14*14
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), window, padding='same'))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))

        #128*14*14 → 64*28*28
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), window, padding='same'))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))

        #64*28*28→32*28*28
        self.G.add(Conv2DTranspose(int(depth/8), window, padding='same'))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))

        #1*28*28
        self.G.add(Conv2DTranspose(output_depth, window, padding='same'))
        #各ピクセルを0～1の間の値にする
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G


    #識別ネットワーク
    #28*28*1の画像が本物かどうかを見分ける
    def discriminator(self, depth=64, dropout=0.3, alpha=0.3):
        if self.D:
            return self.D

        self.D = Sequential()
        input_shape = (self.img_rows, self.img_cols, self.channel)

      #28*28*1 → 14*14*64
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

      #14*14*64 → 7*7*128
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

      #7*7*128 → 4*4*256
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        #4*4*512 → 4*4*512 ####ただしあっているか確認###
        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        #フラット化してsigmoidで分類
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))

        self.D.summary()
        return self.D

    #識別モデル
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(learning_rate=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', \
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    #生成モデル
    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(learning_rate=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', \
                        optimizer=optimizer, metrics=['accuracy'])
        return self.AM
    

## 実際のトレーニングとその結果をアウトプットするクラス.
#MNISTのデータにDCGANを適用するクラス
class MNIST_DCGAN(object):
    #初期化
    def __init__(self, x_train):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = x_train

        #DCGANの識別、敵対的生成モデルの定義
        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    #訓練用の関数
    #train_on_batchは各batchごとに学習している。出力はlossとacc
    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None

        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

        for i in range(train_steps):
            #訓練用のデータをbatch_sizeだけランダムに取り出す
            images_train = self.x_train[np.random.randint(0,self.x_train.shape[0], size=batch_size), :, :, :] 

            # 100*1*1のノイズをbatch sizeだけ生み出して偽画像とする
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

            #生成画像を学習させる
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            #訓練データを1に、生成データを0にする
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0

            #識別モデルを学習させる
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

            #生成&識別モデルを学習させる
            #生成モデルの学習はここでのみ行われる
            a_loss = self.adversarial.train_on_batch(noise, y)

            #訓練データと生成モデルのlossと精度
            #D lossは生成された画像と実際の画像のときのlossとacc
            #A lossはadversarialで生み出された画像を1としたときのlossとacc
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            #save_intervalごとにデータを保存する
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, \
                        samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    #訓練結果をプロットする
    def plot_images(self, save2file=False, fake=True, samples=16, \
                    noise=None, step=0):
        current_path = os.getcwd()
        file = os.path.sep.join(["","data", 'images', 'chapter12', 'synthetic_mnist', ''])
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(current_path+file+filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == "__main__":
    execute_obj = MNIST_DCGAN(x_train)
    execute_obj.train()
    execute_obj.plot_images()