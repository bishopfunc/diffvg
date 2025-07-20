import argparse
import os
from typing import Dict, List, Optional, Tuple

import cma
import matplotlib.pyplot as plt
import numpy as np
import torch

import pydiffvg
from apps.geometry import GeometryLoss


def rect_to_path(rect: pydiffvg.Rect) -> pydiffvg.Path:
    """Rect を Path に変換する。"""
    # 四隅の座標を取得
    x0, y0 = rect.p_min.tolist()
    x1, y1 = rect.p_max.tolist()

    # 四角形の4頂点を定義（時計回り）
    points = torch.tensor(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0],  # 閉じるために始点に戻る
        ],
        dtype=torch.float32,
    )

    # Num control points: 線分だけなので全て0（4つ）
    num_control_points = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

    # Pathオブジェクト作成
    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=points,
        is_closed=True,
        stroke_width=rect.stroke_width,
        id=rect.id,
    )
    return path


def plot_loss_history(
    loss_history_dict: Dict[str, List[float]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """ロス履歴をプロットする関数"""

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


# 引数解析
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", choices=["adam", "cma-es"], default="cma-es")
parser.add_argument("--use-geo-loss", action="store_true", help="幾何学的損失を加える")
parser.add_argument("--exp", type=str, required=True, help="実験名を指定")
args = parser.parse_args()

# 保存先フォルダ
save_dir = f"results/single_rect_{args.exp}"
os.makedirs(save_dir, exist_ok=True)

# デバイス設定
pydiffvg.set_use_gpu(False)
canvas_width, canvas_height = 256, 256

# 目標画像の作成
rect = pydiffvg.Rect(
    p_min=torch.tensor([40.0, 40.0]), p_max=torch.tensor([160.0, 160.0])
)
shapes = [rect]
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

# 幾何損失の初期化
geometry_loss = GeometryLoss(rect_to_path(rect)) if args.use_geo_loss else None


# --- 共通関数 ---
def encode_params(p_min: np.ndarray, p_max: np.ndarray, color: np.ndarray):
    return np.concatenate([p_min / 256.0, p_max / 256.0, color])


def decode_params(params: np.ndarray):
    p_min_n = torch.tensor(params[0:2], dtype=torch.float32)
    p_max_n = torch.tensor(params[2:4], dtype=torch.float32)
    color = torch.tensor(params[4:8], dtype=torch.float32)
    return p_min_n * 256, p_max_n * 256, color


def update_scene(p_min, p_max, color):
    rect.p_min = p_min
    rect.p_max = p_max
    shape_group.fill_color = torch.clamp(color, 0.0, 1.0)


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
    def loss_fn(
        params_np: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        try:
            p_min, p_max, color = decode_params(params_np)
            if (p_min[0] >= p_max[0]) or (p_min[1] >= p_max[1]):
                return 1e6
            update_scene(p_min, p_max, color)
            img = render_only(seed=np.random.randint(10000))
            l2_loss = ((img - target) ** 2).sum()
            geo_loss = torch.tensor(0.0)
            rm_loss = torch.tensor(0.0)
            if args.use_geometry_loss:
                rect = pydiffvg.Rect(p_min=p_min, p_max=p_max)
                geo_loss = geometry_loss.compute(rect_to_path(rect))

            loss = l2_loss + geo_loss + rm_loss
            return loss.item(), l2_loss.item(), geo_loss.item(), rm_loss.item()
        except Exception:
            return 1e6

    initial_params = encode_params(
        np.array([80.0, 20.0]), np.array([100.0, 60.0]), np.array([0.3, 0.2, 0.5, 1.0])
    )
    es = cma.CMAEvolutionStrategy(initial_params, 0.02, {"popsize": 10})
    loss_history = []
    l2_loss_history = []
    geo_loss_history = []
    rm_loss_history = []

    for generation in range(50):
        solutions = es.ask()
        losses = []
        l2_losses = []
        geo_losses = []
        rm_losses = []
        for params in solutions:
            loss, l2_loss, geo_loss, rm_loss = loss_fn(params)
            losses.append(loss)
            l2_losses.append(l2_loss)
            geo_losses.append(geo_loss)
            rm_losses.append(rm_loss)
        es.tell(solutions, losses)
        best_params = es.best.get()[0]
        loss_history.append(min(losses))
        l2_loss_history.append(min(l2_losses))
        geo_loss_history.append(min(geo_losses))
        rm_loss_history.append(min(rm_losses))

        p_min, p_max, color = decode_params(best_params)
        update_scene(p_min, p_max, color)
        render_and_save(
            os.path.join(save_dir, f"iter_{generation}.png"), seed=100 + generation
        )
        print(
            f"Gen {generation}, loss: {min(losses)}",
            f"L2: {min(l2_losses)}, Geo: {min(geo_losses)}, RM: {min(rm_losses)}",
        )
    loss_history_dict = {
        "total_loss": loss_history,
        "l2_loss": l2_loss_history,
        "geometry_loss": geo_loss_history,
        "image_reward_loss": rm_loss_history,
    }
    return loss_history_dict


# --- Adam 最適化 ---
def optimize_adam():
    p_min_n = torch.tensor([80.0 / 256.0, 20.0 / 256.0], requires_grad=True)
    p_max_n = torch.tensor([100.0 / 256.0, 60.0 / 256.0], requires_grad=True)
    color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
    optimizer = torch.optim.Adam([p_min_n, p_max_n, color], lr=1e-2)
    loss_history = []
    geometry_loss_history = []
    l2_loss_history = []
    rm_loss_history = []

    for t in range(50):
        optimizer.zero_grad()
        update_scene(p_min_n * 256, p_max_n * 256, color)
        img = render_and_save(os.path.join(save_dir, f"iter_{t}.png"), seed=100 + t)
        l2_loss = ((img - target) ** 2).sum()
        geo_loss = torch.tensor(0.0)
        rm_loss = torch.tensor(0.0)
        if args.use_geo_loss:
            rect = pydiffvg.Rect(p_min=p_min_n * 256, p_max=p_max_n * 256)
            print(f"{geometry_loss.smooth_nodes=}")
            geo_loss = geometry_loss.compute(rect_to_path(rect))
            print(f"Geometry loss: {geo_loss.item()}")

        loss = l2_loss + geo_loss + rm_loss
        loss.backward()
        optimizer.step()
        l2_loss_history.append(l2_loss.item())
        geometry_loss_history.append(geo_loss.item())
        loss_history.append(loss.item())
        rm_loss_history.append(rm_loss.item())
        print(
            f"Step {t}, loss: {loss.item()}, L2: {l2_loss.item()}, Geo: {geo_loss.item()}, RM: {rm_loss.item()}"
        )

    loss_history_dict = {
        "total_loss": loss_history,
        "l2_loss": l2_loss_history,
        "geometry_loss": geometry_loss_history,
        "image_reward_loss": rm_loss_history,
    }
    return loss_history_dict


# --- 実行 ---
if args.optimizer == "cma-es":
    print("Using CMA-ES optimizer")
    loss_history_dict = optimize_cmaes()
elif args.optimizer == "adam":
    print("Using Adam optimizer")
    loss_history_dict = optimize_adam()
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

# 最終出力・プロット
render_and_save(os.path.join(save_dir, "final.png"), seed=9999)
plot_loss_history(
    loss_history_dict,
    title=f"{args.optimizer.upper()} Loss Curve",
    save_path=os.path.join(save_dir, "loss_plot.png"),
)

# 動画作成（ログ抑制）
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
print(f"Saved results to {save_dir}/out.mp4")
