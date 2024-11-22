import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Activation, Dense, Dropout, Flatten, Conv2D, Conv2DTranspose,
                                     LeakyReLU, Reshape, UpSampling2D, BatchNormalization, Layer)
from tensorflow.keras.optimizers import RMSprop
import sys

# 警告の非表示（任意）
import warnings
warnings.filterwarnings('ignore')

##################################### 学習元データの読み込み #######################################

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

class SelfAttention(Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels

    def build(self, input_shape):
        self.f_conv = Conv2D(self.channels // 8, kernel_size=1, padding='same')
        self.g_conv = Conv2D(self.channels // 8, kernel_size=1, padding='same')
        self.h_conv = Conv2D(self.channels, kernel_size=1, padding='same')
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.softmax = Activation('softmax')

    def call(self, x):
        f = self.f_conv(x)  # Key
        g = self.g_conv(x)  # Query
        h = self.h_conv(x)  # Value

        f_shape = tf.shape(f)
        batch_size, height, width, f_channels = f_shape[0], f_shape[1], f_shape[2], f_shape[3]
        
        f_flat = tf.reshape(f, [batch_size, -1, f_channels])
        g_flat = tf.reshape(g, [batch_size, -1, f_channels])
        h_flat = tf.reshape(h, [batch_size, -1, self.channels])

        attention_map = tf.matmul(g_flat, f_flat, transpose_b=True)
        attention_map = self.softmax(attention_map)

        attention_out = tf.matmul(attention_map, h_flat)
        attention_out = tf.reshape(attention_out, [batch_size, height, width, self.channels])

        out = self.gamma * attention_out + x
        return out

class Zelda_GAN(object):
    def __init__(self, img_rows=12, img_cols=16, channel=8):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None
        self.G = None
        self.AM = None
        self.DM = None

    def generator(self, depth=512, dim=(3, 4), dropout=0.3, momentum=0.8, \
                  window=3, input_dim=32, output_depth=8):
        if self.G:
            return self.G
        self.G = Sequential()

        units = dim[0] * dim[1] * depth
        self.G.add(Dense(units=units, input_dim=input_dim))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim[0], dim[1], depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D())
        self.G.add(Conv2D(256, window, padding='same'))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))
        self.G.add(SelfAttention(256))

        self.G.add(UpSampling2D())
        self.G.add(Conv2D(128, window, padding='same'))
        self.G.add(BatchNormalization(momentum=momentum))
        self.G.add(Activation('relu'))
        self.G.add(SelfAttention(128))

        self.G.add(Conv2D(output_depth, window, padding='same'))
        self.G.add(Activation('softmax'))

        self.G.summary()
        return self.G

    def discriminator(self, depth=64, alpha=0.2, dropout=0.3, window=3):
        if self.D:
            return self.D

        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D = Sequential()

        self.D.add(Conv2D(depth, window, strides=2, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))
        self.D.add(SelfAttention(depth))

        self.D.add(Conv2D(depth * 2, window, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))
        self.D.add(SelfAttention(depth * 2))

        self.D.add(Conv2D(depth * 4, window, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=alpha))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1, activation='sigmoid'))

        self.D.summary()
        return self.D

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(learning_rate=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(learning_rate=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM

class Custom_Zelda_GAN(object):
    def __init__(self, x_train=None, hamming_threshold=10):
        self.img_rows = 12
        self.img_cols = 16
        self.channel = 8
        self.hamming_threshold = hamming_threshold
        if x_train is not None:
            self.x_train = np.transpose(x_train, (0, 2, 3, 1))
            self.x_train = self.x_train.astype('float32') / np.max(self.x_train)
        else:
            self.x_train = None

        self.Zelda_GAN = Zelda_GAN(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel)
        self.discriminator = self.Zelda_GAN.discriminator_model()
        self.adversarial = self.Zelda_GAN.adversarial_model()
        self.generator = self.Zelda_GAN.generator()

        # 重みファイルが存在する場合は読み込む
        self.load_trained_weights()

    def calculate_hamming_distance(self, map1, map2):
        return np.sum(map1 != map2)

    def bootstrap_data(self, generated_maps):
        for generated_map in generated_maps:
            distances = [self.calculate_hamming_distance(generated_map, existing_map) for existing_map in self.x_train]
            if min(distances) > self.hamming_threshold:
                self.x_train = np.append(self.x_train, [generated_map], axis=0)

    def train(self, train_steps=2000, batch_size=32, save_interval=100):
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 32]) if save_interval > 0 else None

        for i in range(train_steps):
            idx = np.random.randint(0, self.x_train.shape[0], size=batch_size)
            images_train = self.x_train[idx]

            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 32])
            images_fake = self.generator.predict(noise)

            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0

            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            a_loss = self.adversarial.train_on_batch(noise, y)

            log_mesg = f"{i}: [D loss: {d_loss[0]}, acc: {d_loss[1]}] [A loss: {a_loss[0]}, acc: {a_loss[1]}]"
            print(log_mesg)

            if (i + 1) % 10 == 0:
                self.bootstrap_data(images_fake)
            
            # 学習経過画像の保存.
            if save_interval > 0 and (i + 1) % save_interval == 0:
                self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))
    
    # 学習の途中経過の画像を保存するメソッド. 白黒なので, わかりやすいようにgenerate_imagesと同様の仕様に修正する.
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
                filename = f"custom_output_{step}.png"
            images = self.generator.predict(noise)
        else:
            idx = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[idx]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i]
            image = np.mean(image, axis=-1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(current_path + file + filename)
            plt.close('all')
        else:
            plt.show()

    def save_trained_weights(self, save_dir="trained_models"):
        # 保存ディレクトリを作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 重みの保存
        generator_path = os.path.join(save_dir, "generator_weights.weights.h5")
        discriminator_path = os.path.join(save_dir, "discriminator_weights.weights.h5")

        self.generator.save_weights(generator_path)
        print(f"Generator weights saved to {generator_path}")

        self.discriminator.save_weights(discriminator_path)
        print(f"Discriminator weights saved to {discriminator_path}")

    # 事前学習済みの重みを読み込むメソッド.
    def load_trained_weights(self, save_dir="trained_models"):
        # 重みファイルが存在する場合は読み込む
        generator_path = os.path.join(save_dir, "generator_weights.weights.h5")
        discriminator_path = os.path.join(save_dir, "discriminator_weights.weights.h5")

        if os.path.exists(generator_path):
            self.generator.load_weights(generator_path)
            print(f"Loaded generator weights from {generator_path}")

        if os.path.exists(discriminator_path):
            self.discriminator.load_weights(discriminator_path)
            print(f"Loaded discriminator weights from {discriminator_path}")

    def generate_images(self, num_images=10):
        # # 複数ページにわたって画像を表示させる.
        # images_per_page = 16
        # num_pages = (num_images + images_per_page - 1) // images_per_page
        # 生成された画像を表示
        noise = np.random.uniform(-1.0, 1.0, size=[num_images, 32])
        images = self.generator.predict(noise)

        # 各チャネルの定義に対応した色を設定
        colors = {
            0: [128, 128, 128],  # 地面 - グレー
            1: [0, 0, 0],        # 壁 - 黒
            2: [255, 215, 0],    # 壺 - ゴールド
            3: [255, 0, 0],      # 敵 - 赤
            4: [0, 255, 0],      # 鍵 - 緑
            5: [0, 0, 255],      # ピックアップアイテム - 青
            6: [0, 255, 255],    # 落とし穴 - シアン
            7: [255, 165, 0]     # 扉 - オレンジ
        }

        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            plt.subplot(4, 4, i + 1)
            image = np.zeros((self.img_rows, self.img_cols, 3), dtype=np.uint8)
            for channel in range(self.channel):
                mask = images[i, :, :, channel] > 0.5  # 閾値を超えたピクセルをそのチャネルのものとみなす
                image[mask] = colors[channel]
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    args = sys.argv
    training_data_path = 'training_data.npy'
    load_weights_only = '--generate_only' in args

    if load_weights_only:
        # 事前学習済みの重みをロードして画像を生成
        gan = Custom_Zelda_GAN()
        gan.generate_images(num_images=10)
    else:
        x_train = load_training_data(training_data_path)
        if x_train is not None:
            gan = Custom_Zelda_GAN(x_train)
            gan.train(train_steps=1000, batch_size=32, save_interval=200)
            gan.save_trained_weights()
