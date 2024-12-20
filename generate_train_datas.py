"""
ステージの要素を0~7の8チャネルでとりあえず表現し,一つのnparray内に対応するチャネル番号を記述してマップを表現する.
これを元に, チャネルごとの12*16のグリッドを自動で作成し, .npy形式で保存してくれる.
出力されたテンソルは, 番号が低い順でイラストのレイヤーのように重ね合わせる仕様でやればいいか. (普通にテンソルの平均を取るのではなく, 数字が一番大きいチャネルをグリッドに適用すればいい)
{地面 : 0, 壁 : 1, プレイヤースタート地点 : 2, 敵 : 3, 鍵 : 4, ピックアップアイテム : 5, 落とし穴 : 6, 扉 : 7}
に変更した.
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import stage_map_arrays as sa

# ぶっちゃけmapデータを読み込む場合は12*16のグリッドを直接チャネル数で埋めて入力しちゃえば良くね？
# 各チャネルの定義は, 今回とりあえず{地面 : 0, 壁 : 1, スタート地点 : 2, 敵 : 3, 鍵 : 4, ピックアップアイテム : 5, 落とし穴 : 6, 扉 : 7}で行こう.
# pythonの多次元リスト表現ってどうなってるんだっけ. もっかい確認しよう.
# 確かサンプルステージから8チャネルのデータを作成してくれるのは確認済み.
# pushするためにコメントアウト追記.

sample_stage_map_list = sa.stage_map_list

# テストに使用するステージ情報を出力するメソッドを作成しよう. 基本的にzelda_ganと同じ画像出力のアプローチでいく.

def generate_and_save_tensor_data(stage_maps, file_path='training_data.npy'):
    """
    ステージのマップ情報に基づいて、(8, 12, 16)のテンソルデータを生成し、ファイルに保存する関数。

    Parameters:
        stage_maps (list of 2D arrays): ステージのマップ情報のリスト。
                                        各マップは形状が (12, 16) の2次元配列。
        file_path (str): 保存するファイルのパス。

    Returns:
        None
    """
    num_samples = len(stage_maps)
    tensor_data = np.zeros((num_samples, 8, 12, 16), dtype=np.float32)

    for i, map_data in enumerate(stage_maps):
        # マップデータが (12, 16) であることを確認
        if map_data.shape != (12, 16):
            raise ValueError(f"マップデータの形状が正しくありません。期待される形状: (12, 16), 実際の形状: {map_data.shape}")

        # マップデータを 8 チャネルのテンソルに変換します。
        # ここでは例として、各セルの値をチャネルに対応させています。
        # 例えば、セルの値が 0 ならチャネル 0 に 1 を設定、他のチャネルは 0 のままにします。

        for y in range(12):
            for x in range(16):
                cell_value = map_data[y, x]
                if 0 <= cell_value < 8:
                    tensor_data[i, int(cell_value), y, x] = 1.0
                else:
                    # セルの値が範囲外の場合はエラーを出力するか、スキップします。
                    pass

    # テンソルデータをファイルに保存します。
    np.save(file_path, tensor_data)
    print(f"テンソルデータが '{file_path}' に保存されました。")


# 学習サンプルを画像として表示するクラス. 学習対象のステージチェックに利用する.
class ZeldaStageVisualizer:
    def __init__(self):
        # 各チャネルの定義に対応した色を設定
        self.colors = {
            0: [128, 128, 128],  # 地面 - グレー
            1: [0, 0, 0],        # 壁 - 黒
            2: [255, 215, 0],    # 壺 - ゴールド
            3: [255, 0, 0],      # 敵 - 赤
            4: [0, 255, 0],      # 鍵 - 緑
            5: [0, 0, 255],      # ピックアップアイテム - 青
            6: [0, 255, 255],    # 落とし穴 - シアン
            7: [255, 165, 0]     # 扉 - オレンジ
        }

    def visualize_stage_maps(self, stage_map_list):
        plt.figure(figsize=(20, 20))
        for idx, stage_map in enumerate(stage_map_list):
            # 35個のマップ表示に対応する.
            plt.subplot(5, 7, idx + 1)
            image = np.zeros((stage_map.shape[0], stage_map.shape[1], 3), dtype=np.uint8)
            for row in range(stage_map.shape[0]):
                for col in range(stage_map.shape[1]):
                    channel = stage_map[row, col]
                    image[row, col] = self.colors.get(channel, [255, 255, 255])  # デフォルトで白
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Stage {idx + 1}')
        plt.tight_layout()
        plt.show()





# 使用例
if __name__ == "__main__":
    # ステージのマップ情報の例を作成します。
    # ここではランダムに 0 から 7 までの整数値を持つマップを 10 個生成します。
    #stage_maps = [np.random.randint(0, 8, size=(12, 16)) for _ in range(10)]
    if len(sys.argv)==1:
        stage_maps = sample_stage_map_list
        #print(f"now data shepe = {stage_maps.shape}")

        # テンソルデータを生成してファイルに保存します。
        generate_and_save_tensor_data(stage_maps, file_path='training_data.npy')

    # 生成したマップグリッドデータの確認.
    elif len(sys.argv)==2 and sys.argv[1] == "conf_data":
        file_path='training_data.npy'
        data = np.load(file_path)
        print(f"data = \n{data}")
        print(f"data shape = {data.shape}")
        # インスタンスを作成してステージ画像を生成する
        visualizer = ZeldaStageVisualizer()
        # ほんとならnpyを読み込んで表示したいけど, 一旦これでやってみよう.
        visualizer.visualize_stage_maps(sample_stage_map_list)
        
