import argparse
import os
from typing import Dict, List, Tuple

import cma
import ImageReward as RM
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import pydiffvg
from apps.geometry import GeometryLoss
from pydiffvg.shape import from_svg_path

RM_SCALE_FACTOR = 1000 # ImageRewardのスコアを100倍するためのスケールファクター
# --- 引数 ---
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", choices=["adam", "cma-es"], default="adam")
parser.add_argument("--use-geo-loss", action="store_true")
parser.add_argument("--use-reward-loss", action="store_true")
parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--iter", type=int, default=50)
args = parser.parse_args()

# --- 初期設定 ---
canvas_width, canvas_height = 510, 510
save_dir = f"results/single_path_{args.exp}"
os.makedirs(save_dir, exist_ok=True)
pydiffvg.set_use_gpu(False)

# --- SVG読み込みとターゲット生成 ---
svg_path_str = "M510,255c0-20.4-17.85-38.25-38.25-38.25H331.5L204,12.75h-51l63.75,204H76.5l-38.25-51H0L25.5,255L0,344.25h38.25l38.25-51h140.25l-63.75,204h51l127.5-204h140.25C492.15,293.25,510,275.4,510,255z"
shapes = from_svg_path(svg_path_str)
path = shapes[0]
path_group = pydiffvg.ShapeGroup(
    shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0])
)
shape_groups = [path_group]
render = pydiffvg.RenderFunction.apply

scene_args = pydiffvg.RenderFunction.serialize_scene(
    canvas_width, canvas_height, shapes, shape_groups
)
target = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args).cpu()
pydiffvg.imwrite(target, os.path.join(save_dir, "target.png"), gamma=2.2)

# --- GeometryLoss 初期化 ---
geometry_loss = GeometryLoss(path) if args.use_geo_loss else None

# --- ImageReward モデルの読み込み ---
model = RM.load("ImageReward-v1.0") if args.use_reward_loss else None
model.eval() if model else None 
prompt = "draw a simple airplane, all lines should be smooth and continuous, no sharp corners or angles, the airplane should be centered in the image, with a clear background."  # 評価プロンプト


# --- utils ---
def render_only(seed: int):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    return render(canvas_width, canvas_height, 2, 2, seed, None, *scene_args).cpu()


def vgimg_to_pilimg(vg_img: torch.Tensor, is_cmaes: bool = False) -> np.ndarray:
    vg_img = (vg_img * 255).clamp(0, 255).byte().cpu().numpy()
    # [H, W, 4] の形状で RGBA 画像と仮定
    pil_img = Image.fromarray(vg_img, mode="RGBA")
    return pil_img  

def render_and_save(filename: str, seed: int):
    img = render_only(seed)
    pydiffvg.imwrite(img, filename, gamma=2.2)
    return img


def encode_params(points: torch.Tensor, color: torch.Tensor) -> np.ndarray:
    return np.concatenate([points.flatten().cpu().numpy() / 510.0, color.cpu().numpy()])


def decode_params(params: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    n = path.points.numel()
    points_flat = params[:n]
    color = params[n:]
    points = (
        torch.tensor(points_flat.reshape(path.points.shape), dtype=torch.float32) * 510
    )
    color = torch.tensor(color, dtype=torch.float32)
    return points, color


# --- Adam 最適化 ---
def optimize_adam():
    points_n = (path.points.clone() + torch.randn_like(path.points) * 20.0) / 510.0
    points_n.requires_grad = True
    color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
    optimizer = torch.optim.Adam([points_n, color], lr=1e-2)
    losses = []
    l2_losses = []
    geo_losses = []
    reward_losses = []

    for t in range(args.iter):
        optimizer.zero_grad()
        path.points = points_n * 510
        path_group.fill_color = color
        img = render_and_save(os.path.join(save_dir, f"iter_{t}.png"), seed=100 + t)

        l2 = ((img - target) ** 2).sum()
        geo = geometry_loss.compute(path) if geometry_loss else torch.tensor(0.0)
        neg_score = torch.tensor(0.0)
        if args.use_reward_loss:
            pil_img = vgimg_to_pilimg(img)
            score = model.score(prompt, pil_img)
            score = torch.tensor(score, requires_grad=False)
            neg_score = -score * RM_SCALE_FACTOR  # スコアをスケールアップ
        loss = l2 + geo + neg_score
        loss.backward()
        optimizer.step()

        print(f"Step {t}, loss={loss.item()}, L2={l2.item()}, Geo={geo.item()}, RM={neg_score.item()}")
        losses.append(loss.item())
        l2_losses.append(l2.item())
        geo_losses.append(geo.item())
        reward_losses.append(neg_score.item())
        loss_history_dict = {
            "total_loss": losses,
            "l2_loss": l2_losses,
            "geometry_loss": geo_losses,
            "image_reward_loss": reward_losses,
        }

    return loss_history_dict


# --- CMA-ES 最適化 ---
def optimize_cmaes():
    init_params = encode_params(path.points, path_group.fill_color)
    es = cma.CMAEvolutionStrategy(init_params, 0.05, {"popsize": 10})
    losses = []
    l2_losses = []
    geo_losses = []
    rm_losses = []

    for g in range(args.iter):
        solutions = es.ask()
        sol_losses = []
        sol_l2_losses = []
        sol_geo_losses = []
        sol_reward_losses = []
        for p in solutions:
            pts, clr = decode_params(p)
            path.points = pts
            path_group.fill_color = clr
            img = render_only(seed=np.random.randint(9999))
            l2 = ((img - target) ** 2).sum()
            geo = geometry_loss.compute(path) if geometry_loss else torch.tensor(0.0)
            neg_score = torch.tensor(0.0)
            if args.use_reward_loss:
                pil_img = vgimg_to_pilimg(img, is_cmaes=True)
                score = model.score(prompt, pil_img)
                score = torch.tensor(score, requires_grad=False)  # 勾配計算から除外
                neg_score = -score * RM_SCALE_FACTOR  # スコアをスケールアップ

            loss = l2 + geo + neg_score
            sol_losses.append(loss.item())
            sol_l2_losses.append(l2.item())
            sol_geo_losses.append(geo.item())
            sol_reward_losses.append(neg_score.item())
        es.tell(solutions, sol_losses)
        best = es.best.get()[0]
        path.points, path_group.fill_color = decode_params(best)
        render_and_save(os.path.join(save_dir, f"iter_{g}.png"), seed=100 + g)
        min_loss_idx = np.argmin(sol_losses)
        losses.append(sol_losses[min_loss_idx])
        l2_losses.append(sol_l2_losses[min_loss_idx])
        geo_losses.append(sol_geo_losses[min_loss_idx])
        rm_losses.append(sol_reward_losses[min_loss_idx])
        print(
            f"Gen {g}, loss={losses[-1]}, L2={l2_losses[-1]}, Geo={geo_losses[-1]}, RM={rm_losses[-1]}"
        )
    loss_history_dict = {
        "total_loss": losses,
        "l2_loss": l2_losses,
        "geometry_loss": geo_losses,
        "image_reward_loss": rm_losses,
    }
    return loss_history_dict


# --- Plot Loss ---
def plot_loss(loss_dict: Dict[str, List[float]], path: str):
    plt.figure()
    for k, v in loss_dict.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(path)


# --- 実行 ---
if args.optimizer == "adam":
    loss_dict = optimize_adam()
else:
    loss_dict = optimize_cmaes()

render_and_save(os.path.join(save_dir, "final.png"), seed=9999)
plot_loss(loss_dict, os.path.join(save_dir, "loss_plot.png"))
print(f"Saved plot to {save_dir}/loss_plot.png")
# --- 動画化 ---
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
print(f"Saved video to {save_dir}/out.mp4")
