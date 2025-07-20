import math
from typing import List, Tuple, Optional, Any, Iterator
import torch

class GeometryLoss:
    """
    SVGパスの幾何学的損失関数を計算するクラスです。
    
    このクラスはパスの幾何学を解析し、以下を含む様々な損失項を計算します:
    - 水平/垂直配置制約
    - 平行線制約
    - パスノードでの滑らかさ制約
    """
    
    def __init__(self, pathObj: Any, xyalign: bool = True, parallel: bool = True, smooth_node: bool = True) -> None:
        """
        パスオブジェクトと制約オプションでGeometryLossを初期化します。
        
        引数:
            pathObj: 点と制御点情報を含むパスオブジェクト
            xyalign: 水平/垂直配置制約を適用するかどうか
            parallel: 平行線制約を適用するかどうか
            smooth_node: ノードでの滑らかさ制約を適用するかどうか
        """
        self.pathObj = pathObj
        self.pathId = pathObj.id
        
        # パスからセグメントを抽出
        self.get_segments(pathObj)
        
        # 制約フラグを初期化
        self.xyalign = xyalign
        self.parallel = parallel
        self.smooth_node = smooth_node
        
        # 有効な場合、水平/垂直制約を作成
        if xyalign:
            self.make_hor_ver_constraints(pathObj)

        # 有効な場合、平行線制約を作成
        if parallel:
            self.make_parallel_constraints(pathObj)

        # 有効な場合、滑らかさ制約を作成
        if smooth_node:
            self.make_smoothness_constraints(pathObj)

    def make_smoothness_constraints(self, pathObj: Any) -> None:
        """
        パス内の滑らかなノードを識別し、それらの接線長比を保存します。
        
        入力接線と出力接線の間の角度が非常に小さい（< 1e-2）場合、
        ノードは滑らかであると見なされます。
        
        引数:
            pathObj: 点と制御点情報を含むパスオブジェクト
        """
        self.smooth_nodes: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[float, float]]] = []
        
        for idx, node in enumerate(self.iterate_nodes()):
            # 滑らかさと接線ベクトルを計算
            sm, t0, t1 = self.node_smoothness(node, pathObj)
            
            if abs(sm) < 1e-2:
                # ノードが滑らか - ノード情報と正規化された接線長を保存
                tangent_length_ratio_0 = (t0.norm() / self.segment_approx_length(node[0], pathObj)).item()
                tangent_length_ratio_1 = (t1.norm() / self.segment_approx_length(node[1], pathObj)).item()
                self.smooth_nodes.append((node, (tangent_length_ratio_0, tangent_length_ratio_1)))

    def node_smoothness(self, node: Tuple[Tuple[int, int], Tuple[int, int]], pathObj: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        接線間の角度を測定してノードの滑らかさを計算します。
        
        滑らかさは、出力接線と回転した入力接線の内積を
        それらの大きさで正規化して計算されます。
        
        引数:
            node: (前のセグメント, 次のセグメント)情報を含むタプル
            pathObj: 点と制御点情報を含むパスオブジェクト
            
        戻り値:
            (滑らかさ値, 出力接線, 入力接線)のタプル
        """
        # ノードでの接線ベクトルを取得
        t0 = self.tangent_out(node[0], pathObj)  # 出力接線
        t1 = self.tangent_in(node[1], pathObj)   # 入力接線
        
        # 外積計算のために入力接線を90度回転
        t1rot = torch.stack((-t1[1], t1[0]))
        
        # 正規化された内積を計算（滑らかさの測定値）
        smoothness = t0.dot(t1rot) / (t0.norm() * t1.norm())

        return smoothness, t0, t1

    def segment_approx_length(self, segment: Tuple[int, int], pathObj: Any) -> torch.Tensor:
        """
        パスセグメントの近似長を計算します。
        
        直線の場合: 端点間の直接距離
        二次曲線の場合: 制御多角形の辺の長さの合計
        三次曲線の場合: 制御多角形の辺の長さの合計
        
        引数:
            segment: (セグメント種別, セグメントインデックス)のタプル
                    セグメント種別は 0=直線, 1=二次, 2=三次
            pathObj: 点と制御点情報を含むパスオブジェクト
            
        戻り値:
            セグメントの近似長をtorch.Tensorとして返す
        """
        if segment[0] == 0:
            # 直線セグメント - 端点間の直接距離
            idxs = self.segList[segment[0]][segment[1]]
            length = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]).norm()
            return length
        elif segment[0] == 1:
            # 二次ベジエ曲線 - 制御多角形の辺の長さの合計
            idxs = self.segList[segment[0]][segment[1]]
            length = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]).norm() + \
                    (pathObj.points[idxs[2], :] - pathObj.points[idxs[1], :]).norm()
            return length
        elif segment[0] == 2:
            # 三次ベジエ曲線 - 制御多角形の辺の長さの合計
            idxs = self.segList[segment[0]][segment[1]]
            length = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]).norm() + \
                    (pathObj.points[idxs[2], :] - pathObj.points[idxs[1], :]).norm() + \
                    (pathObj.points[idxs[3], :] - pathObj.points[idxs[2], :]).norm()
            return length
        
        # ここに到達することはないはず
        raise ValueError(f"Unknown segment type: {segment[0]}")

    def tangent_in(self, segment: Tuple[int, int], pathObj: Any) -> torch.Tensor:
        """
        セグメントの開始点での入力接線ベクトルを計算します。
        
        引数:
            segment: (セグメント種別, セグメントインデックス)のタプル
                    セグメント種別は 0=直線, 1=二次, 2=三次
            pathObj: 点と制御点情報を含むパスオブジェクト
            
        戻り値:
            入力接線ベクトルをtorch.Tensorとして返す
        """
        if segment[0] == 0:
            # 直線セグメント - 接線は線の方向の半分
            idxs = self.segList[segment[0]][segment[1]]
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :]) / 2
            return tangent
        elif segment[0] == 1:
            # 二次ベジエ - 接線は開始点から最初の制御点へ
            idxs = self.segList[segment[0]][segment[1]]
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :])
            return tangent
        elif segment[0] == 2:
            # 三次ベジエ - 接線は開始点から最初の制御点へ
            idxs = self.segList[segment[0]][segment[1]]
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[0], :])
            return tangent

        raise ValueError(f"Unknown segment type: {segment[0]}")

    def tangent_out(self, segment: Tuple[int, int], pathObj: Any) -> torch.Tensor:
        """
        セグメントの終了点での出力接線ベクトルを計算します。
        
        引数:
            segment: (セグメント種別, セグメントインデックス)のタプル
                    セグメント種別は 0=直線, 1=二次, 2=三次
            pathObj: 点と制御点情報を含むパスオブジェクト
            
        戻り値:
            出力接線ベクトルをtorch.Tensorとして返す
        """
        if segment[0] == 0:
            # 直線セグメント - 接線は逆方向の線の半分
            idxs = self.segList[segment[0]][segment[1]]
            tangent = (pathObj.points[idxs[0], :] - pathObj.points[idxs[1], :]) / 2
            return tangent
        elif segment[0] == 1:
            # 二次ベジエ - 接線は最後の制御点から終了点へ
            idxs = self.segList[segment[0]][segment[1]]
            tangent = (pathObj.points[idxs[1], :] - pathObj.points[idxs[2], :])
            return tangent
        elif segment[0] == 2:
            # 三次ベジエ - 接線は最後から2番目の制御点から終了点へ
            idxs = self.segList[segment[0]][segment[1]]
            tangent = (pathObj.points[idxs[2], :] - pathObj.points[idxs[3], :])
            return tangent

        raise ValueError(f"Unknown segment type: {segment[0]}")

    def get_segments(self, pathObj: Any) -> None:
        """
        パスオブジェクトからパスセグメントを抽出し分類します。
        
        セグメントは以下のように分類されます:
        - 直線 (制御点0個): self.linesに保存
        - 二次ベジエ曲線 (制御点1個): self.quadricsに保存
        - 三次ベジエ曲線 (制御点2個): self.cubicsに保存
        
        引数:
            pathObj: 点と制御点情報を含むパスオブジェクト
        """
        # セグメント格納を初期化
        self.segments: List[Tuple[int, int]] = []
        self.lines: List[Tuple[int, ...]] = []
        self.quadrics: List[Tuple[int, ...]] = []
        self.cubics: List[Tuple[int, ...]] = []
        self.segList = (self.lines, self.quadrics, self.cubics)
        
        idx = 0
        total_points = pathObj.points.shape[0]
        
        # 制御点数に基づいて各セグメントを処理
        for ncp in pathObj.num_control_points.numpy():
            if ncp == 0:
                # 直線セグメント（制御点なし）
                self.segments.append((0, len(self.lines)))
                self.lines.append((idx, (idx + 1) % total_points))
                idx += 1
            elif ncp == 1:
                # 二次ベジエ曲線（制御点1個）
                self.segments.append((1, len(self.quadrics)))
                self.quadrics.append((idx, (idx + 1), (idx + 2) % total_points))
                idx += ncp + 1
            elif ncp == 2:
                # 三次ベジエ曲線（制御点2個）
                self.segments.append((2, len(self.cubics)))
                self.cubics.append((idx, (idx + 1), (idx + 2), (idx + 3) % total_points))
                idx += ncp + 1

    def iterate_nodes(self) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        パス内のすべてのノードを反復処理します。
        
        ノードは隣接する2つのセグメント間の接続点として定義されます。
        各ノードは(前のセグメント, 次のセグメント)のタプルとして表現されます。
        
        戻り値:
            各ノードに対して(前のセグメント, 次のセグメント)のタプルを生成
        """
        # 隣接するセグメントのペアを作成、最後を最初に接続するためにラップアラウンド
        for prev, next in zip([self.segments[-1]] + self.segments[:-1], self.segments):
            yield (prev, next)

    def make_hor_ver_constraints(self, pathObj: Any) -> None:
        """
        配置制約のための水平および垂直線セグメントを識別します。
        
        x方向の差が < 1e-6 の場合、線は水平と見なされ、
        y方向の差が < 1e-6 の場合、線は垂直と見なされます。
        
        引数:
            pathObj: 点と制御点情報を含むパスオブジェクト
        """
        self.horizontals: List[int] = []
        self.verticals: List[int] = []
        
        for idx, line in enumerate(self.lines):
            # 線の開始点と終了点を取得
            startPt = pathObj.points[line[0], :]
            endPt = pathObj.points[line[1], :]
            
            # 差分ベクトルを計算
            dif = endPt - startPt
            
            # 線が水平かチェック（x方向の差が最小）
            if abs(dif[0]) < 1e-6:
                self.horizontals.append(idx)
            
            # 線が垂直かチェック（y方向の差が最小）
            if abs(dif[1]) < 1e-6:
                self.verticals.append(idx)

    def make_parallel_constraints(self, pathObj: Any) -> None:
        """
        平行制約のために類似した傾きを持つ線セグメントをグループ化します。
        
        傾きが1e-3ラジアン以内の線は平行と見なされます。
        xyalignが有効な場合、水平線と垂直線は除外されます。
        
        引数:
            pathObj: 点と制御点情報を含むパスオブジェクト
        """
        slopes: List[Tuple[float, List[int]]] = []
        
        for lidx, line in enumerate(self.lines):
            # 線の端点を取得
            startPt = pathObj.points[line[0], :]
            endPt = pathObj.points[line[1], :]
            
            # 方向ベクトルを計算
            dif = endPt - startPt
            
            # 傾き角度を計算（[0, π]に正規化）
            slope = math.atan2(dif[1], dif[0])
            if slope < 0:
                slope += math.pi
            
            # 許容範囲内の既存の傾きグループを検索
            minidx = -1
            for idx, s in enumerate(slopes):
                if abs(s[0] - slope) < 1e-3:
                    minidx = idx
                    break
            
            # 既存のグループに追加するか新しいグループを作成
            if minidx >= 0:
                slopes[minidx][1].append(lidx)
            else:
                slopes.append((slope, [lidx]))
        
        # グループをフィルタ: 1本以上の線が必要、xyalignが有効な場合は水平/垂直を除外
        self.parallel_groups: List[List[int]] = [
            sgroup[1] for sgroup in slopes 
            if len(sgroup[1]) > 1 and (
                not self.xyalign or (
                    sgroup[0] > 1e-3 and abs(sgroup[0] - (math.pi / 2)) > 1e-3
                )
            )
        ]

    def make_line_diff(self, pathObj, lidx):
        """
        線セグメントの方向ベクトルを計算します。
        
        引数:
            pathObj: パスオブジェクト
            lidx: 線のインデックス
            
        戻り値:
            方向ベクトル
        """
        line = self.lines[lidx]
        startPt = pathObj.points[line[0], :]
        endPt = pathObj.points[line[1], :]

        dif = endPt - startPt
        return dif

    def calc_hor_ver_loss(self, loss, pathObj):
        """
        水平・垂直配置制約の損失を計算します。
        
        引数:
            loss: 累積損失値
            pathObj: パスオブジェクト
        """
        # 水平線の損失を計算
        for lidx in self.horizontals:
            dif = self.make_line_diff(pathObj, lidx)
            loss += dif[0].pow(2)

        # 垂直線の損失を計算
        for lidx in self.verticals:
            dif = self.make_line_diff(pathObj, lidx)
            loss += dif[1].pow(2)

    def calc_parallel_loss(self, loss, pathObj):
        """
        平行線制約の損失を計算します。
        
        引数:
            loss: 累積損失値
            pathObj: パスオブジェクト
        """
        for group in self.parallel_groups:
            # グループ内の各線の方向ベクトルを取得
            diffs = [self.make_line_diff(pathObj, lidx) for lidx in group]
            difmat = torch.stack(diffs, 1)
            
            # 長さを計算して正規化
            lengths = difmat.pow(2).sum(dim=0).sqrt()
            difmat = difmat / lengths
            
            # 回転行列を作成して外積を計算
            difmat = torch.cat((difmat, torch.zeros(1, difmat.shape[1])))
            rotmat = difmat[:, list(range(1, difmat.shape[1])) + [0]]
            cross = difmat.cross(rotmat)
            
            # 平行線損失を計算
            ploss = cross.pow(2).sum() * lengths.sum() * 10
            loss += ploss

    def calc_smoothness_loss(self, loss, pathObj):
        """
        滑らかさ制約の損失を計算します。
        
        引数:
            loss: 累積損失値
            pathObj: パスオブジェクト
        """
        for node, tlengths in self.smooth_nodes:
            # ノードの滑らかさを計算
            sl, t0, t1 = self.node_smoothness(node, pathObj)
            
            # 滑らかさ損失を追加
            loss += sl.pow(2) * t0.norm().sqrt() * t1.norm().sqrt()
            
            # 接線長比の損失を計算
            tl = ((t0.norm() / self.segment_approx_length(node[0], pathObj)) - tlengths[0]).pow(2) + \
                 ((t1.norm() / self.segment_approx_length(node[1], pathObj)) - tlengths[1]).pow(2)
            loss += tl * 10

    def compute(self, pathObj):
        """
        幾何学的制約に基づく総損失を計算します。
        
        引数:
            pathObj: パスオブジェクト
            
        戻り値:
            計算された総損失
        """
        # パスIDの一致を確認
        if pathObj.id != self.pathId:
            raise ValueError("Path ID {} does not match construction-time ID {}".format(pathObj.id, self.pathId))

        # 損失を初期化
        loss = torch.tensor(0.)
        
        # 水平/垂直配置制約の損失を計算
        if self.xyalign:
            self.calc_hor_ver_loss(loss, pathObj)

        # 平行線制約の損失を計算
        if self.parallel:
            self.calc_parallel_loss(loss, pathObj)

        # 滑らかさ制約の損失を計算
        if self.smooth_node:
            self.calc_smoothness_loss(loss, pathObj)

        return loss
