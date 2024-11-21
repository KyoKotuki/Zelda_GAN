## GVGAI Zeldaのステージ生成用GAN. 動くかどうかを試してみる.
## サンプルGANの実装には次のQiita(https://qiita.com/takuto512/items/c8fd8eb1cdc7b6689798)を参考にした.
## 繰り返し似たようなメソッドや引数が出てくるので, 一度勉強したらわりかし楽かもしれん.

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Activation, Dense, Dropout, Flatten, Conv2D, Conv2DTranspose,
                                     LeakyReLU, Reshape, UpSampling2D, add, BatchNormalization, Input, Lambda, Layer)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import ZeroPadding2D

# 警告の非表示（任意）
import warnings
warnings.filterwarnings('ignore')

##################################### 学習元データの読み込み #######################################

# 外部ファイルから学習データを読み込む関数を定義します。
def load_training_data(file_path):
    """
    学習用のデータを外部ファイルから読み込む関数。
    データは NumPy の .npy ファイルとして保存されているものとします。
    """
    if os.path.exists(file_path):
        x_train = np.load(file_path)
        print(f"Training data loaded successfully from {file_path}.")
        return x_train
    else:
        print(f"Training data file not found at {file_path}.")
        return None

#######################################################################################################

# self-Attention実装クラス. レイヤー内でラムダ式で複雑なレイヤーを呼び出すことはできないので, クラスとして個別に実装する.
class SelfAttention(Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels

    def build(self, input_shape):
        # 学習可能なレイヤーをここで定義
        self.f_conv = Conv2D(self.channels // 8, kernel_size=1, padding='same')
        self.g_conv = Conv2D(self.channels // 8, kernel_size=1, padding='same')
        self.h_conv = Conv2D(self.channels, kernel_size=1, padding='same')
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.softmax = Activation('softmax')

    def call(self, x):
        # 1x1の畳み込みでキー、クエリ、バリューを作成
        f = self.f_conv(x)  # Key
        g = self.g_conv(x)  # Query
        h = self.h_conv(x)  # Value

        # 形状を (batch_size, height*width, channels) に変換
        f_shape = tf.shape(f)
        batch_size = f_shape[0]
        height = f_shape[1]
        width = f_shape[2]
        f_channels = f_shape[3]

        # Reshape for matrix multiplication
        f_flat = tf.reshape(f, [batch_size, -1, f_channels])  # (batch_size, height*width, channels//8)
        g_flat = tf.reshape(g, [batch_size, -1, f_channels])  # (batch_size, height*width, channels//8)
        h_flat = tf.reshape(h, [batch_size, -1, self.channels])  # (batch_size, height*width, channels)

        # Attention Mapの計算
        attention_map = tf.matmul(g_flat, f_flat, transpose_b=True)  # (batch_size, height*width, height*width)
        attention_map = self.softmax(attention_map)  # 正規化

        # Attentionの適用
        attention_out = tf.matmul(attention_map, h_flat)  # (batch_size, height*width, channels)
        attention_out = tf.reshape(attention_out, [batch_size, height, width, self.channels])  # 元の形状に戻す

        # 出力を計算
        out = self.gamma * attention_out + x

        return out

## Zelda_GANのクラス定義
class Zelda_GAN(object):
    # 初期化
    def __init__(self, img_rows=12, img_cols=16, channel=8):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # Generatorのモデル定義
    def generator(self, depth=512, dim=(3, 4), dropout=0.3, momentum=0.8, \
                window=3, input_dim=32, output_depth=8):
        if self.G:
            return self.G
        self.G = Sequential()

        # 全結合層でランダムノイズを変換
        units = dim[0] * dim[1] * depth  # units = 3 * 4 * 512 = 6,144
        self.G.add(Dense(units=units, input_dim=input_dim))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim[0], dim[1], depth)))  # Reshape to (3, 4, 512)
        self.G.add(Dropout(dropout))

        # アップサンプリングと畳み込み
        self.G.add(UpSampling2D())  # サイズを2倍に（6, 8, 512）
        self.G.add(Conv2D(256, window, padding='same'))  # フィルター数を256に
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))

        # Self-Attention Block
        self.G.add(SelfAttention(256))

        # アップサンプリングと畳み込み
        self.G.add(UpSampling2D())  # サイズを2倍に（12, 16, 256）
        self.G.add(Conv2D(128, window, padding='same'))  # フィルター数を128に
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))

        # Self-Attention Block
        self.G.add(SelfAttention(128))

        # 出力層
        self.G.add(Conv2D(output_depth, window, padding='same'))  # フィルター数を8に
        self.G.add(Activation('softmax'))

        self.G.summary()
        return self.G


    # Discriminatorのモデル定義
    def discriminator(self, depth=64, alpha=0.2, dropout=0.3, window=3):
        if self.D:
            return self.D
        
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D = Sequential()

        # 畳み込み層
        self.D.add(Conv2D(depth, window, strides=2, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        # Self-Attention Block
        #self.D.add(Lambda(lambda x: self.self_attention(x, depth)))
        self.D.add(SelfAttention(depth))

        # 畳み込み層
        self.D.add(Conv2D(depth * 2, window, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        # Self-Attention Block
        #self.D.add(Lambda(lambda x: self.self_attention(x, depth * 2)))
        self.D.add(SelfAttention(depth*2))

        # 畳み込み層
        self.D.add(Conv2D(depth * 4, window, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        # 出力層
        self.D.add(Flatten())
        self.D.add(Dense(1, activation='sigmoid'))

        self.D.summary()
        return self.D

    # 識別モデル
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(learning_rate=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', \
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    # 敵対的生成モデル
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

## 実際のトレーニングとその結果をアウトプットするクラス
class Custom_Zelda_GAN(object):
    # 初期化
    def __init__(self, x_train):
        self.img_rows = 12  # 高さ
        self.img_cols = 16  # 幅
        self.channel = 8    # チャンネル数 (8)

        self.x_train = x_train

        # データの形状を調整： (サンプル数, 8, 12, 16) -> (サンプル数, 12, 16, 8)
        self.x_train = np.transpose(self.x_train, (0, 2, 3, 1))

        # 正規化（0から1の範囲に）
        self.x_train = self.x_train.astype('float32') / np.max(self.x_train)

        # Zelda_GANの識別、敵対的生成モデルの定義
        self.Zelda_GAN = Zelda_GAN(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel)
        self.discriminator = self.Zelda_GAN.discriminator_model()
        self.adversarial = self.Zelda_GAN.adversarial_model()
        self.generator = self.Zelda_GAN.generator()

    # 訓練用の関数
    def train(self, train_steps=2000, batch_size=32, save_interval=100):
        noise_input = None

        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 32])  # Generatorの入力ノイズベクトル

        for i in range(train_steps):
            # 訓練用のデータをbatch_sizeだけランダムに取り出す
            idx = np.random.randint(0, self.x_train.shape[0], size=batch_size)
            images_train = self.x_train[idx]

            # ランダムノイズを生成
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 32])

            # 生成画像を生成
            images_fake = self.generator.predict(noise)

            # 訓練データと生成データを結合
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0  # 生成データのラベルを0に

            # 識別モデルを学習
            d_loss = self.discriminator.train_on_batch(x, y)

            # Generatorの学習
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 32])

            # 敵対的生成モデルを学習
            a_loss = self.adversarial.train_on_batch(noise, y)

            # ログの出力
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            # 生成画像の保存
            if save_interval > 0 and (i + 1) % save_interval == 0:
                self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))

    # 訓練結果をプロットする関数
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        current_path = os.getcwd()
        file = os.path.sep.join(["", "generated_images", ""])
        if not os.path.exists(current_path + file):
            os.makedirs(current_path + file)
        filename = 'custom_output.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "custom_output_%d.png" % step
            images = self.generator.predict(noise)
        else:
            idx = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[idx]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i]
            # チャンネルごとの平均を取る（視覚化のため）
            image = np.mean(image, axis=-1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(current_path + file + filename)
            plt.close('all')
        else:
            plt.show()
    
    # グレースケールだとわかりずらいので, 各出力についてチャネル毎の画像を出力し, 一括でディレクトリ内に保存するメソッドを作成する.
    def plot_each_channels(self):
        pass

    # 毎回トレーニングするのはややこしいので, 一旦学習させたモデルは保存して, 後からロードして利用できるようにしたい.
    # モデルを保存するメソッド.
    def save_trained_weights(self, save_dir="trained_models"):
        # 保存ディレクトリを作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 重みの保存
        generator_path = os.path.join(save_dir, "generator_weights.h5")
        discriminator_path = os.path.join(save_dir, "discriminator_weights.h5")
        
        self.generator.save_weights(generator_path)
        print(f"Generator weights saved to {generator_path}")
        
        self.discriminator.save_weights(discriminator_path)
        print(f"Discriminator weights saved to {discriminator_path}")

    
    # 待て, 学習モデルを保存することができても, 出力された画像(テンソル)を保存しておかないとゲーム制作に使えない.
    # Playabilityを考慮して, 利用可能なマップは貯蔵するようにしなければ. 実行ファイルをmain.pyとかにして別に作ろうかな. 管理しきれなくなるわ.

if __name__ == "__main__":
    # 学習データのパスを指定
    training_data_path = 'training_data.npy'  # ここを実際のデータファイルのパスに変更してください

    # 学習データの読み込み
    x_train = load_training_data(training_data_path)
    if x_train is None:
        print("No training data available. Exiting.")
    else:
        # モデルの作成と訓練
        gan = Custom_Zelda_GAN(x_train)
        gan.train(train_steps=1000, batch_size=32, save_interval=200)

        # 訓練済みモデルの保存.
        gan.save_trained_models(save_dir="trained_models")
