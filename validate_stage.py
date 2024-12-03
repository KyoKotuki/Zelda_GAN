"""
生成されたステージについて, 制約条件を満たしているかを判定するメソッド群.
今のところはメソッドのみの実装. 必要があればクラスにする.
{地面 : 0, 壁 : 1, プレイヤースタート地点 : 2, 敵 : 3, 鍵 : 4, ピックアップアイテム : 5, 落とし穴 : 6, 扉 : 7}
"""
from collections import deque
import numpy as np

# selfが使えないからもうクラスにしちゃうわ.
class ValidateStage():
    def validate_stages(self, stage):
        """
        ステージが指定する制約を満たしているかを確認する関数.
        - プレイヤー, 鍵, ゴールマス(扉)は1つずつ存在する.
        - ダイヤモンドマスは10個以上存在する.
        - プレイヤマスから鍵およびゴールマスへ到達可能である.
        - ステージの上下左右は壁マスに囲まれている.
        """
        player_count = np.sum(stage == 2)  # プレイヤースタート地点: チャネル番号 2
        key_count = np.sum(stage == 4)  # 鍵: チャネル番号 4
        goal_count = np.sum(stage == 7)  # ゴールマス (扉): チャネル番号 7
        diamond_count = np.sum(stage == 5)  # ピックアップアイテム (ダイヤモンド): チャネル番号 5

        if player_count != 1 or key_count != 1 or goal_count != 1:
            return False

        if diamond_count < 10:
            return False

        if not self.is_reachable(stage, 2, 4) or not self.is_reachable(stage, 2, 7):
            return False

        if not self.is_surrounded_by_walls(stage):
            return False

        return True

    def is_reachable(self, stage, start_value, target_value):
        """
        プレイヤーマスからゴールマスへ到達可能であるかを確認する関数.
        BFSを使用して到達可能性をチェックしている.
        """
        start_pos = tuple(np.argwhere(stage == start_value)[0])
        queue = deque([start_pos])
        visited = set([start_pos])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            if stage[x, y] == target_value:
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < stage.shape[0] and 0 <= ny < stage.shape[1] and (nx, ny) not in visited:
                    if stage[nx, ny] != 1:  # 1は壁マス
                        queue.append((nx, ny))
                        visited.add((nx, ny))

        return False  # ゴールに到達できない場合

    def is_surrounded_by_walls(self, stage):
        """
        ステージの上下左右が壁マス（値が1）で囲まれているかを確認する関数。
        """
        if not np.all(stage[0, :] == 1):  # 上辺
            return False
        if not np.all(stage[-1, :] == 1):  # 下辺
            return False
        if not np.all(stage[:, 0] == 1):  # 左辺
            return False
        if not np.all(stage[:, -1] == 1):  # 右辺
            return False
        return True
