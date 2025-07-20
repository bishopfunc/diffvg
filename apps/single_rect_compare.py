import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch

import pydiffvg
from apps.geometry import GeometryLoss


# --- ベジエ曲線の Path を作成 ---
def make_cubic_path():
    points = torch.tensor(
        [
            [40.0, 40.0],  # p0
            [80.0, 40.0],  # c1
            [120.0, 160.0],  # c2
            [160.0, 160.0],  # p1
        ],
        dtype=torch.float32,
    )

    num_control_points = torch.tensor([2], dtype=torch.int32)  # cubic
    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=points,
        is_closed=False,
        stroke_width=torch.tensor(2.0),
        id=0,
    )
    return path


def plot_loss_history(
    loss_history_dict: Dict[str, List[float]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(8, 5))
    for key, losses in loss_history_dict.items():
        plt.plot(losses, marker="o", label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title or "Optimization Loss Curve")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()


# --- 引数処理 ---
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", choices=["adam", "cma-es"], default="cma-es")
parser.add_argument("--use-geo-loss", action="store_true")
parser.add_argument("--exp", type=str, required=True)
args = parser.parse_args()

save_dir = f"results/bezier_path_{args.exp}"
os.makedirs(save_dir, exist_ok=True)

pydiffvg.set_use_gpu(False)
canvas_width, canvas_height = 256, 256

# --- Path 作成とターゲット画像生成 ---
path = make_cubic_path()
shapes = [path]
shape_group = pydiffvg.ShapeGroup(
    shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0])
)
shape_groups = [shape_group]
render = pydiffvg.RenderFunction.apply
scene_args = pydiffvg.RenderFunction.serialize_scene(
    canvas_width, canvas_height, shapes, shape_groups
)
target = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args).cpu()
pydiffvg.imwrite(target, os.path.join(save_dir, "target.png"), gamma=2.2)

# --- GeometryLoss の初期化 ---
geometry_loss = GeometryLoss(path) if args.use_geo_loss else None

# --- 最適化対象のパラメータ（点群）を学習対象に ---
points = path.points.clone()
points.requires_grad = True
optimizer = torch.optim.Adam([points], lr=1e-1)

loss_history, geo_loss_history, l2_loss_history = [], [], []

# --- 最適化ループ（Adam） ---
for t in range(50):
    optimizer.zero_grad()
    path.points = points
    img = render(
        canvas_width,
        canvas_height,
        2,
        2,
        t + 10,
        None,
        *pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        ),
    ).cpu()
    l2_loss = ((img - target) ** 2).sum()
    geo = geometry_loss.compute(path) if args.use_geo_loss else torch.tensor(0.0)
    loss = l2_loss + geo
    loss.backward()
    optimizer.step()
    pydiffvg.imwrite(img, os.path.join(save_dir, f"iter_{t}.png"), gamma=2.2)
    loss_history.append(loss.item())
    l2_loss_history.append(l2_loss.item())
    geo_loss_history.append(geo.item())
    print(
        f"Step {t:02d}: total={loss.item():.2f}, l2={l2_loss.item():.2f}, geo={geo.item():.5f}"
    )

# --- ロスプロットと動画保存 ---
plot_loss_history(
    {"total": loss_history, "l2": l2_loss_history, "geo": geo_loss_history},
    title="Bezier Path Optimization",
    save_path=os.path.join(save_dir, "loss_curve.png"),
)

from subprocess import DEVNULL, call

call(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        "24",
        "-i",
        os.path.join(save_dir, "iter_%d.png"),
        "-vb",
        "20M",
        os.path.join(save_dir, "out.mp4"),
    ],
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print(f"Saved result to {save_dir}/out.mp4")
