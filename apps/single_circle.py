import pydiffvg
import torch
import skimage
import numpy as np
import pathlib

# 必要なライブラリをインポート
# pydiffvg: 微分可能なベクターグラフィックスライブラリ
# torch: PyTorchテンソル操作とオプティマイザー用
# skimage, numpy: 画像処理用
# pathlib: ファイルパス操作用

# GPUが利用可能な場合は使用（現在は無効化）
# pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_use_gpu(False)

# キャンバスサイズを設定
canvas_width = 256
canvas_height = 256

# 目標となる円を作成（半径: 40, 中心: [128, 128]）
circle = pydiffvg.Circle(radius = torch.tensor(40.0),
                         center = torch.tensor([128.0, 128.0]))
shapes = [circle]

# 円の描画グループを作成（緑色で塗りつぶし）
circle_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
    fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [circle_group]

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
             None,
             *scene_args)

# 出力画像はリニアRGB空間なので、保存前にガンマ補正を適用
save_path = pathlib.Path('results/single_circle/target.png')
save_path.parent.mkdir(parents=True, exist_ok=True)  # ディレクトリを作成
pydiffvg.imwrite(img.cpu(), save_path, gamma=2.2)
target = img.clone()  # 目標画像として保存

# 初期推定値として円を移動
# 学習率を調整しやすくするため半径と中心を正規化
radius_n = torch.tensor(20.0 / 256.0, requires_grad=True)
center_n = torch.tensor([108.0 / 256.0, 138.0 / 256.0], requires_grad=True)
color = torch.tensor([0.3, 0.2, 0.8, 1.0], requires_grad=True)  # 青色に変更

# 正規化された値をピクセル座標に戻す
circle.radius = radius_n * 256
circle.center = center_n * 256
circle_group.fill_color = color

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

# 初期画像を保存
pydiffvg.imwrite(img.cpu(), 'results/single_circle/init.png', gamma=2.2)

# 半径と中心、色を最適化
optimizer = torch.optim.Adam([radius_n, center_n, color], lr=1e-2)
# Adamオプティマイザーで100回反復実行
# 最適化ループ
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    
    # フォワードパス: 画像を描画
    circle.radius = radius_n * 256
    circle.center = center_n * 256
    circle_group.fill_color = color
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
    pydiffvg.imwrite(img.cpu(), 'results/single_circle/iter_{}.png'.format(t), gamma=2.2)
    
    # 損失関数を計算（L2ノルム）
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # 勾配を逆伝播
    loss.backward()
    
    # 勾配を表示
    print('radius.grad:', radius_n.grad)
    print('center.grad:', center_n.grad)
    print('color.grad:', color.grad)

    # 勾配降下ステップを実行
    optimizer.step()
    
    # 現在のパラメータを表示
    print('radius:', circle.radius)
    print('center:', circle.center)
    print('color:', circle_group.fill_color)

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
pydiffvg.imwrite(img.cpu(), 'results/single_circle/final.png')

# 中間描画結果を動画に変換
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_circle/iter_%d.png", "-vb", "20M",
    "results/single_circle/out.mp4"])
