import pydiffvg
import torch
import skimage
import numpy as np

# 必要なライブラリをインポート
# pydiffvg: 微分可能なベクターグラフィックスライブラリ
# torch: PyTorchテンソル操作とオプティマイザー用
# skimage, numpy: 画像処理用

# GPUが利用可能な場合は使用する設定（現在は無効化）
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_use_gpu(False)  # GPUを使わない設定

# キャンバスサイズを設定
canvas_width, canvas_height = 256 ,256

# 目標となる矩形を作成（左上: [40, 40], 右下: [160, 160]）
rect = pydiffvg.Rect(p_min = torch.tensor([40.0, 40.0]),
                     p_max = torch.tensor([160.0, 160.0]))
shapes = [rect]

# 矩形の描画グループを作成（緑色で塗りつぶし）
rect_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [rect_group]

# シーンをシリアライズして描画準備
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

# 描画関数を取得
render = pydiffvg.RenderFunction.apply

# 目標画像を描画
img = render(256, # 幅
             256, # 高さ
             2,   # x方向サンプル数
             2,   # y方向サンプル数
             0,   # 乱数シード
             None, # 背景画像
             *scene_args)

# 出力画像はリニアRGB空間なので、保存前にガンマ補正を適用
pydiffvg.imwrite(img.cpu(), 'results/single_rect/target.png', gamma=2.2)
target = img.clone()  # 目標画像として保存

# 初期推定値として矩形を移動
# 学習率を調整しやすくするためp_min & p_maxを正規化
p_min_n = torch.tensor([80.0 / 256.0, 20.0 / 256.0], requires_grad=True)
p_max_n = torch.tensor([100.0 / 256.0, 60.0 / 256.0], requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)  # 紫色に変更
# 正規化された値をピクセル座標に戻す
rect.p_min = p_min_n * 256
rect.p_max = p_max_n * 256
rect_group.fill_color = color

# 初期状態のシーンを準備
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

# 初期画像を描画
img = render(256, # 幅
             256, # 高さ
             2,   # x方向サンプル数
             2,   # y方向サンプル数
             1,   # 乱数シード
             None, # 背景画像
             *scene_args)

# 初期画像を保存
pydiffvg.imwrite(img.cpu(), 'results/single_rect/init.png', gamma=2.2)

# 矩形の位置とサイズ、色を最適化
optimizer = torch.optim.Adam([p_min_n, p_max_n, color], lr=1e-2)
# Adamオプティマイザーで100回反復実行
# 最適化ループ
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    
    # フォワードパス: 画像を描画
    rect.p_min = p_min_n * 256
    rect.p_max = p_max_n * 256
    rect_group.fill_color = color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # 幅
                 256,   # 高さ
                 2,     # x方向サンプル数
                 2,     # y方向サンプル数
                 t+1,   # 乱数シード（反復ごとに変更）
                 None, # 背景画像
                 *scene_args)
    
    # 中間結果を保存
    pydiffvg.imwrite(img.cpu(), 'results/single_rect/iter_{}.png'.format(t), gamma=2.2)
    
    # 損失関数を計算（L2ノルム）
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # 勾配を逆伝播
    loss.backward()
    
    # 勾配を表示
    print('p_min.grad:', p_min_n.grad)
    print('p_max.grad:', p_max_n.grad)
    print('color.grad:', color.grad)

    # 勾配降下ステップを実行
    optimizer.step()
    
    # 現在のパラメータを表示
    print('p_min:', rect.p_min)
    print('p_max:', rect.p_max)
    print('color:', rect_group.fill_color)

# 最終結果を描画
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # 幅
             256,   # 高さ
             2,     # x方向サンプル数
             2,     # y方向サンプル数
             102,    # 乱数シード
             None, # 背景画像
             *scene_args)

# 最終画像を保存
pydiffvg.imwrite(img.cpu(), 'results/single_rect/final.png')

# 中間描画結果を動画に変換
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_rect/iter_%d.png", "-vb", "20M",
    "results/single_rect/out.mp4"])
