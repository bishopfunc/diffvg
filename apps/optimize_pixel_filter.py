"""
ピクセルフィルター最適化デモ

このスクリプトはピクセルフィルターの半径パラメータを最適化して、
目標画像に近づけるデモンストレーションです。
Hannフィルターの半径を勾配降下法で調整し、
目標画像との差を最小化します。
"""

import diffvg
import pydiffvg
import torch
import skimage
import numpy as np

# GPUが利用可能な場合は使用
pydiffvg.set_use_gpu(torch.cuda.is_available())

# キャンバスサイズを設定
canvas_width = 256
canvas_height = 256

# 目標となる円を作成（半径40、中心[128, 128]）
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]))
shapes = [circle]

# 円の描画グループを作成（緑色で塗りつぶし）
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [circle_group]

# 目標画像用のシーンを準備（Hannフィルター、半径8.0）
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    shapes=shapes,
    shape_groups=shape_groups,
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = torch.tensor(8.0)))

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
pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/target.png', gamma=2.2)
target = img.clone()  # 目標画像として保存

# 初期推定値としてピクセルフィルター半径を変更（1.0に設定）
radius = torch.tensor(1.0, requires_grad = True)

# 初期状態のシーンを準備（最適化対象の半径を使用）
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    shapes=shapes,
    shape_groups=shape_groups,
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = radius))

# 初期画像を描画
img = render(256, # 幅
             256, # 高さ
             2,   # x方向サンプル数
             2,   # y方向サンプル数
             1,   # 乱数シード
             None,
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/init.png', gamma=2.2)

# フィルター半径の最適化を実行
optimizer = torch.optim.Adam([radius], lr=1.0)

# Adamオプティマイザーで100回反復実行
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    
    # フォワードパス: 画像を描画
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        shapes=shapes,
        shape_groups=shape_groups,
        filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                    radius = radius))
    img = render(256,   # 幅
                 256,   # 高さ
                 2,     # x方向サンプル数
                 2,     # y方向サンプル数
                 t+1,   # 乱数シード（反復ごとに変更）
                 None,
                 *scene_args)
    
    # 中間結果を保存
    pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/iter_{}.png'.format(t), gamma=2.2)
    
    # 損失関数を計算（L2ノルム）
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # 勾配を逆伝播
    loss.backward()
    
    # 勾配を表示
    print('radius.grad:', radius.grad)

    # 勾配降下ステップを実行
    optimizer.step()
    
    # 現在のパラメータを表示
    print('radius:', radius)

# 最終結果を描画
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width=canvas_width,
    canvas_height=canvas_height,
    shapes=shapes,
    shape_groups=shape_groups,
    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.hann,
                                radius = radius))
img = render(256,   # 幅
             256,   # 高さ
             2,     # x方向サンプル数
             2,     # y方向サンプル数
             102,    # 乱数シード
             None,
             *scene_args)

# 最終画像を保存
pydiffvg.imwrite(img.cpu(), 'results/optimize_pixel_filter/final.png')

# 中間描画結果を動画に変換
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/optimize_pixel_filter/iter_%d.png", "-vb", "20M",
    "results/optimize_pixel_filter/out.mp4"])
