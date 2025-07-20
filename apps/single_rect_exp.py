import argparse
import os
from typing import List, Optional

import cma
import matplotlib.pyplot as plt
import numpy as np
import torch

import pydiffvg
from apps.geometry import GeometryLoss

# 引数解析
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", choices=["adam", "cma-es"], default="cma-es")
parser.add_argument(
    "--use-geometry-loss", action="store_true", help="幾何学的損失を加える"
)
parser.add_argument("--exp", type=str, required=True, help="実験名を指定")
args = parser.parse_args()

# 出力先ディレクトリを exp 変数で管理
exp = f"results/{args.exp}"
os.makedirs(exp, exist_ok=True)


# --- プロット関数 ---
def plot_loss_history(
    loss_history: List[float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker="o", label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title or "Optimization Loss Curve")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss curve to {save_path}")
    else:
        plt.show()


# --- 初期設定 ---
pydiffvg.set_use_gpu(False)
canvas_width, canvas_height = 256, 256


# パス生成

path = pydiffvg.Path(
    num_control_points=torch.tensor(
        [0]
    ),  # 各セグメントの制御点数（0: 直線, 1: 二次ベジエ, 2: 三次ベジエ）
    points=torch.tensor(
        [[80.0, 20.0], [100.0, 60.0]]
    ),  # パスのすべての点（端点・制御点）
    stroke_width=torch.tensor(1.0),  # 線の太さ（fill_colorを使うなら使われない）
    is_closed=False,  # 閉じたパスかどうか
)

path.id = 0  # GeometryLoss に必要

shapes = [path]
shape_group = pydiffvg.ShapeGroup(
    shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.3, 0.2, 0.5, 1.0])
)
shape_groups = [shape_group]
render = pydiffvg.RenderFunction.apply

# ターゲット画像を描画
target_path = pydiffvg.Path(
    num_control_points=torch.tensor([0]),
    points=torch.tensor([[40.0, 40.0], [160.0, 160.0]]),
    stroke_width=torch.tensor(1.0),
    is_closed=False,
)
target_path.id = 0
target_shapes = [target_path]
target_group = pydiffvg.ShapeGroup(
    shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0])
)
target_shape_groups = [target_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(
    canvas_width, canvas_height, target_shapes, [target_group]
)
target = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args).cpu()
pydiffvg.imwrite(target, os.path.join(exp, "target.png"), gamma=2.2)

# 幾何損失の初期化
geometry_loss = GeometryLoss(path) if args.use_geometry_loss else None


# --- 共通関数 ---
def update_scene():
    shapes[0] = path
    shape_groups[0] = shape_group


def render_only(seed: int):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    return render(canvas_width, canvas_height, 2, 2, seed, None, *scene_args).cpu()


def render_and_save(filename: str, seed: int):
    img = render_only(seed)
    pydiffvg.imwrite(img, filename, gamma=2.2)
    return img


# --- CMA-ES 最適化 ---
def optimize_cmaes():
    def encode_params():
        return np.concatenate(
            [
                path.points.view(-1).detach().numpy(),
                shape_group.fill_color.detach().numpy(),
            ]
        )

    def decode_params(params: np.ndarray):
        n_pts = path.points.numel()
        path.points.data = torch.tensor(
            params[:n_pts].reshape(-1, 2), dtype=torch.float32
        )
        shape_group.fill_color.data = torch.tensor(
            params[n_pts : n_pts + 4], dtype=torch.float32
        )

    init_params = encode_params()
    es = cma.CMAEvolutionStrategy(init_params, 5.0, {"popsize": 10})
    loss_history = []

    for t in range(50):
        solutions = es.ask()
        losses = []
        for s in solutions:
            decode_params(s)
            update_scene()
            img = render_only(seed=np.random.randint(10000))
            loss = ((img - target) ** 2).sum()
            if geometry_loss:
                loss += geometry_loss.compute(path)
            losses.append(loss.item())
        es.tell(solutions, losses)
        decode_params(es.best.get()[0])
        update_scene()
        render_and_save(os.path.join(exp, f"iter_{t}.png"), 100 + t)
        print(f"Gen {t}, loss: {min(losses)}")
        loss_history.append(min(losses))
    return loss_history


# --- Adam 最適化 ---
def optimize_adam():
    path.points.requires_grad = True
    shape_group.fill_color.requires_grad = True
    optimizer = torch.optim.Adam([path.points, shape_group.fill_color], lr=1e-1)
    loss_history = []

    for t in range(50):
        optimizer.zero_grad()
        update_scene()
        img = render_and_save(os.path.join(exp, f"iter_{t}.png"), 100 + t)
        loss = ((img - target) ** 2).sum()
        if geometry_loss:
            loss += geometry_loss.compute(path)
        loss.backward()
        optimizer.step()
        print(f"Step {t}, loss: {loss.item()}")
        loss_history.append(loss.item())
    return loss_history


# --- 実行 ---
if args.optimizer == "cma-es":
    print("Using CMA-ES optimizer")
    loss_history = optimize_cmaes()
else:
    print("Using Adam optimizer")
    loss_history = optimize_adam()

# --- 結果保存 ---
render_and_save(os.path.join(exp, "final.png"), seed=9999)
plot_loss_history(
    loss_history,
    title=f"{args.optimizer.upper()} Loss",
    save_path=os.path.join(exp, "loss_plot.png"),
)

# --- 動画生成（ログなし）---
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
        os.path.join(exp, "iter_%d.png"),
        "-vb",
        "20M",
        os.path.join(exp, "out.mp4"),
    ],
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print(f"Saved results to {exp}/out.mp4")
