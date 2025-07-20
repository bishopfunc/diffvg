"""
二次距離近似の比較デモ

このスクリプトは二次ベジエ曲線の距離場計算において、
正確な距離計算と二次近似の比較を行います。
SDFと通常のカラー描画の両方で結果を可視化し、
近似手法の精度を評価します。
"""

import pydiffvg
import torch
import skimage
import numpy as np
import matplotlib.pyplot as plt

# GPUが利用可能な場合は使用
pydiffvg.set_use_gpu(torch.cuda.is_available())

# キャンバスサイズを設定
canvas_width, canvas_height = 256, 256

# 二次ベジエ曲線の制御点を定義
num_control_points = torch.tensor([1])  # 1つの制御点（二次ベジエ）
points = torch.tensor([[ 50.0,  30.0], # 開始点
                       [125.0, 400.0], # 制御点
                       [170.0,  30.0]]) # 終了点

# ストローク付きのパスを作成（距離近似は無効）
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     stroke_width = torch.tensor([30.0]),
                     is_closed = False,
                     use_distance_approx = False)  # 正確な距離計算を使用
shapes = [path]

# パスの描画グループを作成（グレーのストローク）
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = None,
                                 stroke_color = torch.tensor([0.5, 0.5, 0.5, 0.5]))
shape_groups = [path_group]

# === 正確な距離場の計算 ===

# SDF出力用のシーンを準備
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)

render = pydiffvg.RenderFunction.apply

# SDF画像を描画
img = render(256, # 幅
             256, # 高さ
             1,   # x方向サンプル数
             1,   # y方向サンプル数
             0,   # 乱数シード
             None, # 背景画像
             *scene_args)

# SDF値を正規化してカラーマップを適用
img /= 256.0
cm = plt.get_cmap('viridis')
img = cm(img.squeeze())
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/ref_sdf.png')

# 通常のカラー描画
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # 幅
             256, # 高さ
             2,   # x方向サンプル数
             2,   # y方向サンプル数
             0,   # 乱数シード
             None, # 背景画像
             *scene_args)
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/ref_color.png')

# === 二次近似距離場の計算 ===

# 距離近似を有効化
shapes[0].use_distance_approx = True

# SDF出力用のシーンを再準備
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups,
    output_type = pydiffvg.OutputType.sdf)

# 近似SDF画像を描画
img = render(256, # 幅
             256, # 高さ
             1,   # x方向サンプル数
             1,   # y方向サンプル数
             0,   # 乱数シード
             None, # 背景画像
             *scene_args)

# SDF値を正規化してカラーマップを適用
img /= 256.0
img = cm(img.squeeze())
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/approx_sdf.png')

# 近似を使った通常のカラー描画
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # 幅
             256, # 高さ
             2,   # x方向サンプル数
             2,   # y方向サンプル数
             0,   # 乱数シード
             None, # 背景画像
             *scene_args)
pydiffvg.imwrite(img, 'results/quadratic_distance_approx/approx_color.png')
