import pathlib

import skimage
import torch
from torchvision import transforms

import pydiffvg


# GPUが使える場合は使う
# pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_use_gpu(False)

# 目標画像を読み込む
target_path = pathlib.Path("results/font_sample/target.png")
target = skimage.io.imread(target_path)
print(f"target.shape = {target.shape}")

target = torch.from_numpy(target).to(torch.float32) / 255.0  # [0,1] に正規化
target = target.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
# サイズを変更
target = transforms.functional.resize(
    target, (256, 256), antialias=True
)  # サイズを256x256に変更
target = target.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
print(f"target.shape = {target.shape}")

# 条件設定
canvas_width, canvas_height = 256, 256  # 画像サイズは256x256

# ベジェ曲線のパラメータ
num_control_points = torch.tensor(
    [2, 2, 2]
)  # 3ノードの閉じたベジェ曲線で、各点ごとに2個の制御点を持つ想定
scale = torch.tensor(
    [[canvas_width, canvas_height]], dtype=torch.float32
)  # 正規化を元に戻すときに使うスケール

# 制御点パラメータ：最適化対象
# 初期値は正解値を少しずらしたものになっている
# 変数が典型的な値におさまるよう正規化している
points_n = torch.nn.Parameter(
    torch.tensor(
        [
            [100.0 / canvas_width, 40.0 / canvas_height],  # base
            [155.0 / canvas_width, 65.0 / canvas_height],  # control point
            [100.0 / canvas_width, 180.0 / canvas_height],  # control point
            [65.0 / canvas_width, 238.0 / canvas_height],  # base
            [100.0 / canvas_width, 200.0 / canvas_height],  # control point
            [170.0 / canvas_width, 55.0 / canvas_height],  # control point
            [220.0 / canvas_width, 100.0 / canvas_height],  # base
            [210.0 / canvas_width, 80.0 / canvas_height],  # control point
            [140.0 / canvas_width, 60.0 / canvas_height],
        ]
    )
)  # control point
# 色パラメータ：最適化対象
color = torch.nn.Parameter(torch.tensor([0.3, 0.2, 0.5, 1.0]))

# 引数を用意
path = pydiffvg.Path(
    num_control_points=num_control_points, points=points_n * scale, is_closed=True
)
shapes = [path]  # 形状のリスト：ラスタライズ時の引数になる。
path_group = pydiffvg.ShapeGroup(
    shape_ids=torch.tensor([0]),  # shapes内のインデックスを指定する
    fill_color=color,
)  # グループでまとめて色を指定できる
shape_groups = [path_group]  # 形状グループのリスト：ラスタライズ時の引数になる

print(f"points_n.shape = {points_n.shape}")
print(f"shapes.shape = {[shape.points.shape for shape in shapes]}")
print(f"shape_groups.shape = {[group.fill_color.shape for group in shape_groups]}")


# Optimizer指定
optimizer = torch.optim.Adam([points_n, color], lr=0.02)
# 最適化ループ
for t in range(100):
    optimizer.zero_grad()

    # パラメータ再設定
    # 各iterationで再設定が必要。
    path.points = points_n * scale
    path_group.fill_color = color

    # 引数の前処理
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    # ラスタライズ実行
    # 内部の近似で乱数を使っているためnum_samples_x/y, seedを指定する。num_samples_x/yは後続研究でも2を使っているものが殆ど。
    img = pydiffvg.RenderFunction.apply(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        t + 1,  # seed
        None,  # background_image
        *scene_args,
    )

    # 損失関数を計算（平均二乗誤差）
    loss = (img - target).pow(2).mean()

    # 動作確認用
    print("loss:", loss.item())
    pydiffvg.imwrite(img.cpu(), f"results/font_sample/iter_{t:02d}.png", gamma=2.2)

    # バックプロパゲーション
    loss.backward()

    # パラメータ更新
    optimizer.step()


from subprocess import call

call(
    [
        "ffmpeg",
        "-framerate",
        "24",
        "-i",
        "results/font_sample/iter_%02d.png",
        "-vb",
        "20M",
        "results/font_sample/out.mp4",
    ]
)
