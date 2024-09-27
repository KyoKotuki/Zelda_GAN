## GVGAI Zeldaのステージ生成用GAN. 動くかどうかを試してみる.
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


#################################################################################

# データセット入力. 手動でテンソルを作成することになるのかな？
# テストデータとトレーニングデータに分けないといけないよな. 元論文だったらどうやってるんだ？
channels = 8
rows = 12
cols = 16
# 生成するステージデータ数.
samples = 8

# 8*12*16のテンソルをsampleで指定した数だけ生成. あとは手動で設定する必要がある.
tensor = np.zeros((samples, channels, rows, cols))

##################################################################################



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

    # SelfAttention層. 元論文内の設計では利用されていたため, 手動で実装.
    def self_attention(x, channels):
        f = Conv2D(channels // 8, kernel_size=1)(x)
        g = Conv2D(channels // 8, kernel_size=1)(x)
        h = Conv2D(channels, kernel_size=1)(x)

        attention_map = tf.matmul(Flatten(f), Flatten(g), transpose_b=True)
        attention_map = Activation('softmax')(attention_map)

        attention = tf.matmul(attention_map, Flatten(h))
        attention = Reshape(tf.shape(x))(attention)

        return add()([attention, x])

    
    # Generatorのモデル定義
    def generator(self, depth=512, dim=3, dropout=0.3, momentum=0.8, \
                  window=3, input_dim=32, output_depth=8):
        if self.G:
            return self.G
        self.G = Sequential()

        #input_layer = Input(shape=(32,))  # 入力は (32,) のベクトル
        # 32 ⇨ 512*3*4の全結合層で変換.
        self.G.add(Dense(dim*dim*depth, input_dim=input_dim))
        # バッチ正規化. 具体的な値はどう設定されているのかは論文から推察するのは難しいかも.
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, 4, depth))) # Reshape to (512, 3, 4)
        self.G.add(Dropout(dropout))

        ## 以下, 工事中 ##

        # Layer 1 - Deconvolution, Batch Normalization, ReLU
        x = Dense(512 * 3 * 4)(input_layer)
        x = Reshape((512, 3, 4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Layer 2 - Upsample, Convolution, Batch Normalization, ReLU
        x = UpSampling2D()(x)
        x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Layer 3 - Self-Attention
        x = self.self_attention(x, 256)  # Self-Attention Layer

        # Layer 4 - Upsample, Convolution, Batch Normalization, ReLU
        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Layer 5 - Self-Attention
        x = self.self_attention(x, 128)  # Self-Attention Layer

        # Layer 6 - Convolution, Softmax
        x = Conv2D(8, kernel_size=(3, 3), padding='same')(x)
        output_layer = Activation('softmax')(x)

        self.G = Model(input_layer, output_layer)
        self.G.summary()
        return self.G
    
    # Discriminatorのモデル定義
    # 8*12*16の画像が本物かどうかを見分ける. ← 次元の順番はこれであってるのだろうか.
    def discriminator(self):
        if self.D:
            return self.D

        input_layer = Input(shape=(8, 12, 16))  # ステージのサイズ

        # Layer 1 - Convolution, LeakyReLU
        x = Conv2D(128, kernel_size=(3, 3), padding='same')(input_layer)
        x = LeakyReLU(alpha=0.2)(x)

        # Layer 2 - Self-Attention
        x = self.self_attention(x, 128)  # Self-Attention Layer

        # Layer 3 - Convolution, LeakyReLU
        x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Layer 4 - Self-Attention
        x = self.self_attention(x, 256)  # Self-Attention Layer

        # Layer 5 - Convolution, LeakyReLU
        x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Layer 6 - Convolution, Flatten, Sigmoid
        x = Conv2D(1, kernel_size=(3, 3), padding='same')(x)
        x = Flatten()(x)
        output_layer = Activation('sigmoid')(x)

        self.D = Model(input_layer, output_layer)
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
# (8, 12, 16)のテンソルデータにDCGANを適用するクラス
class Custom_DCGAN(object):
    # 初期化
    def __init__(self, x_train):
        self.img_rows = 12  # 高さ
        self.img_cols = 16  # 幅
        self.channel = 8    # チャンネル数 (8)

        self.x_train = x_train

        # DCGANの識別、敵対的生成モデルの定義
        self.DCGAN = DCGAN()
        self.discriminator = self.DCGAN.discriminator_model()  # Discriminatorモデル
        self.adversarial = self.DCGAN.adversarial_model()      # Adversarialモデル
        self.generator = self.DCGAN.generator()                # Generatorモデル

    # 訓練用の関数
    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None

        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 32])  # Generatorの入力ノイズベクトル (16, 32)

        for i in range(train_steps):
            # 訓練用のデータをbatch_sizeだけランダムに取り出す
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]

            # 32次元のランダムノイズを生成
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 32])  # Generatorの入力ノイズベクトル (batch_size, 32)

            # 生成画像を学習させる
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            # 訓練データを1に、生成データを0にする
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0

            # 識別モデルを学習させる
            d_loss = self.discriminator.train_on_batch(x, y)

            # Generatorの学習
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 32])

            # 生成&識別モデルを学習させる
            a_loss = self.adversarial.train_on_batch(noise, y)

            # 訓練データと生成モデルのlossと精度
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            # save_intervalごとにデータを保存する
            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))

    # 訓練結果をプロットする関数
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        current_path = os.getcwd()
        file = os.path.sep.join(["", "data", 'images', 'custom', ''])
        filename = 'custom_output.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 32])
            else:
                filename = "custom_output_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, 0]  # 一つのチャンネルを表示
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(current_path + file + filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == "__main__":
    pass

