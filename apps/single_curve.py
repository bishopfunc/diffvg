"""
単一ベジエ曲線の最適化デモ

このスクリプトは複数のベジエ曲線セグメントで構成される閉じたパスを作成し、
目標形状に向けて曲線の制御点と色を最適化します。
勾配降下法を使用して、目標画像との差を最小化します。
"""

import pydiffvg
import torch
import skimage
import numpy as np

# GPUが利用可能な場合は使用
pydiffvg.set_use_gpu(torch.cuda.is_available())

# キャンバスサイズを設定
canvas_width, canvas_height = 256, 256

# 各セグメントの制御点数を定義（3つのセグメント、各2つの制御点）
num_control_points = torch.tensor([2, 2, 2])

# 目標となるベジエ曲線の制御点を定義
points = torch.tensor([[120.0,  30.0], # 基点
                       [150.0,  60.0], # 制御点
                       [ 90.0, 198.0], # 制御点
                       [ 60.0, 218.0], # 基点
                       [ 90.0, 180.0], # 制御点
                       [200.0,  65.0], # 制御点
                       [210.0,  98.0], # 基点
                       [220.0,  70.0], # 制御点
                       [130.0,  55.0]]) # 制御点

# 閉じたパスを作成
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = True)
shapes = [path]

# パスの描画グループを作成（緑色で塗りつぶし）
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]

# シーンをシリアライズして描画準備
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

# 目標画像を描画
render = pydiffvg.RenderFunction.apply
img = render(256, # 幅
             256, # 高さ
             2,   # x方向サンプル数
             2,   # y方向サンプル数
             0,   # 乱数シード
             None,
             *scene_args)

# 出力画像はリニアRGB空間なので、保存前にガンマ補正を適用
pydiffvg.imwrite(img.cpu(), 'results/single_curve/target.png', gamma=2.2)
target = img.clone()  # 目標画像として保存

# 初期推定値として曲線を移動
# 学習率を調整しやすくするため制御点を正規化
points_n = torch.tensor([[100.0/256.0,  40.0/256.0], # 基点
                         [155.0/256.0,  65.0/256.0], # 制御点
                         [100.0/256.0, 180.0/256.0], # 制御点
                         [ 65.0/256.0, 238.0/256.0], # 基点
                         [100.0/256.0, 200.0/256.0], # 制御点
                         [170.0/256.0,  55.0/256.0], # 制御点
                         [220.0/256.0, 100.0/256.0], # 基点
                         [210.0/256.0,  80.0/256.0], # 制御点
                         [140.0/256.0,  60.0/256.0]], # 制御点
                        requires_grad = True) 

# 初期色を設定（紫色）
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)

# 正規化された値をピクセル座標に戻す
path.points = points_n * 256
path_group.fill_color = color

# 初期状態のシーンを準備
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

# 初期画像を描画
img = render(256, # 幅
             256, # 高さ
             2,   # x方向サンプル数
             2,   # y方向サンプル数
             1,   # 乱数シード
             None,
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_curve/init.png', gamma=2.2)

# 最適化の実行
optimizer = torch.optim.Adam([points_n, color], lr=1e-2)

# Adamオプティマイザーで100回反復実行
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    
    # フォワードパス: 画像を描画
    path.points = points_n * 256
    path_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # 幅
                 256,   # 高さ
                 2,     # x方向サンプル数
                 2,     # y方向サンプル数
                 t+1,   # 乱数シード（反復ごとに変更）
                 None,
                 *scene_args)
    
    # 中間結果を保存
    pydiffvg.imwrite(img.cpu(), 'results/single_curve/iter_{}.png'.format(t), gamma=2.2)
    
    # 損失関数を計算（L2ノルム）
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # 勾配を逆伝播
    loss.backward()
    
    # 勾配を表示
    print('points_n.grad:', points_n.grad)
    print('color.grad:', color.grad)

    # 勾配降下ステップを実行
    optimizer.step()
    
    # 現在のパラメータを表示
    print('points:', path.points)
    print('color:', path_group.fill_color)

# 最終結果を描画
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # 幅
             256,   # 高さ
             2,     # x方向サンプル数
             2,     # y方向サンプル数
             102,    # 乱数シード
             None,
             *scene_args)

# 最終画像を保存
pydiffvg.imwrite(img.cpu(), 'results/single_curve/final.png')

# 中間描画結果を動画に変換
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_curve/iter_%d.png", "-vb", "20M",
    "results/single_curve/out.mp4"])
