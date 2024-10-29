"""
ステージの要素を0~7の8チャネルでとりあえず表現し,一つのnparray内に対応するチャネル番号を記述してマップを表現する.
これを元に, チャネルごとの12*16のグリッドを自動で作成し, .npy形式で保存してくれる.
"""
import numpy as np
import sys

# ぶっちゃけmapデータを読み込む場合は12*16のグリッドを直接チャネル数で埋めて入力しちゃえば良くね？
# 各チャネルの定義は, 今回とりあえず{0 : 地面, 1 : 壁, 2 : 壺, 3 : 敵, 4 : 鍵, 5 : ピックアップアイテム, 6 : 落とし穴, 7 : 扉}で行こう.
# pythonの多次元リスト表現ってどうなってるんだっけ. もっかい確認しよう.
# 確かサンプルステージから8チャネルのデータを作成してくれるのは確認済み.

sample_stage_map_list = [
    np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1],
    [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=int),
    np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=int)
]

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