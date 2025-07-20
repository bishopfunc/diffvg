import argparse
import os
from typing import List, Optional

import cma
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
# from apps.geometry import GeometryLoss
import pydiffvg
# from apps.geometry import GeometryLoss
import ImageReward as RM


def plot_loss_history(
    loss_history: List[float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """ロス履歴をプロットする関数"""

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker="o", label="Best loss per iteration")
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
import matplotlib
matplotlib.use('Agg')

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
# 背景合成処理を追加
target = target[:, :, 3:4] * target[:, :, :3] + torch.ones(target.shape[0], target.shape[1], 3) * (1 - target[:, :, 3:4])
target = target.detach().cpu()
pydiffvg.imwrite(target, os.path.join(save_dir, "target.png"), gamma=1.0)

# 幾何損失の初期化
# geometry_loss = GeometryLoss(rect) if args.use_geo_loss else None

model = RM.load("ImageReward-v1.0")
model.eval()  # 評価モードに設定
prompt = "smooth rectangle shape"  # 評価プロンプト

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
    # 背景合成処理を追加
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3) * (1 - img[:, :, 3:4])
    # 保存前に明示的にCPUに移動
    img_cpu_for_save = img.detach().cpu()
    pydiffvg.imwrite(img_cpu_for_save, filename, gamma=1.0)
    return img


# --- CMA-ES 最適化 ---
def optimize_cmaes():
    def loss_fn(params_np: np.ndarray):
        try:
            p_min, p_max, color = decode_params(params_np)
            if (p_min[0] >= p_max[0]) or (p_min[1] >= p_max[1]):
                return 1e6
            update_scene(p_min, p_max, color)
            img = render_only(seed=np.random.randint(10000))
            # 背景合成処理を追加
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3) * (1 - img[:, :, 3:4])
            l2_loss = ((img - target) ** 2).sum()
            
            # ImageReward用の画像前処理
            try:
                # 画像を0-255の範囲に変換してPIL Imageとして準備
                img_for_reward = (img * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                from PIL import Image
                pil_img = Image.fromarray(img_for_reward)
                score = model.score(prompt, pil_img)
            except Exception as e:
                print(f"ImageReward error: {e}")
                score = 0.0  # エラー時はスコアを0にする
            
            if args.use_geo_loss:
                geo_loss = torch.tensor(0.0)
            else:
                geo_loss = torch.tensor(0.0)
            loss = l2_loss + geo_loss - score
            return loss.item()
        except Exception as e:
            print(f"Error in loss function with params: {params_np}, error: {e}")
            return 1e6

    initial_params = encode_params(
        np.array([80.0, 20.0]), np.array([100.0, 60.0]), np.array([0.3, 0.2, 0.5, 1.0])
    )
    es = cma.CMAEvolutionStrategy(initial_params, 0.02, {"popsize": 10})
    loss_history = []

    for generation in range(50):
        solutions = es.ask()
        losses = [loss_fn(s) for s in solutions]
        es.tell(solutions, losses)
        best_params = es.best.get()[0]
        loss_history.append(min(losses))

        p_min, p_max, color = decode_params(best_params)
        update_scene(p_min, p_max, color)
        render_and_save(
            os.path.join(save_dir, f"iter_{generation}.png"), seed=100 + generation
        )
        print(f"Gen {generation}, loss: {min(losses)}")

    return loss_history


# --- Adam 最適化 ---
def optimize_adam():
    p_min_n = torch.tensor([80.0 / 256.0, 20.0 / 256.0], requires_grad=True)
    p_max_n = torch.tensor([100.0 / 256.0, 60.0 / 256.0], requires_grad=True)
    color = torch.tensor([0.3, 0.2, 0.5, 1.0], requires_grad=True)
    optimizer = torch.optim.Adam([p_min_n, p_max_n, color], lr=1e-2)
    loss_history = []
    geometry_loss_history = []
    l2_loss_history = []

    for t in range(50):  # デバッグのために少なくする
        optimizer.zero_grad()
        update_scene(p_min_n * 256, p_max_n * 256, color)
        img = render_and_save(os.path.join(save_dir, f"iter_{t}.png"), seed=100 + t)
        # imgは既に背景合成済みなのでそのまま使用
        l2_loss = ((img - target) ** 2).sum()
        
        # ImageReward用の画像前処理（Adam版）
        try:
            img_for_reward = (img.detach().cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_for_reward)
            score = model.score(prompt, pil_img)
            score = torch.tensor(score, requires_grad=False)  # 勾配計算から除外
        except Exception as e:
            print(f"ImageReward error in Adam: {e}")
            score = torch.tensor(0.0, requires_grad=False)
        
        if args.use_geo_loss:
            geo_loss = torch.tensor(0.0)  # geometry_lossは現在無効
        else:
            geo_loss = torch.tensor(0.0)
        loss = l2_loss + geo_loss - score
        loss.backward()
        optimizer.step()
        l2_loss_history.append(l2_loss.item())
        geometry_loss_history.append(geo_loss.item())
        loss_history.append(loss.item())
        print(f"Step {t}, loss: {loss.item()}")

    return loss_history


# --- 実行 ---
if args.optimizer == "cma-es":
    print("Using CMA-ES optimizer")
    loss_history = optimize_cmaes()
elif args.optimizer == "adam":
    print("Using Adam optimizer")
    loss_history = optimize_adam()
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

# 最終出力・プロット
try:
    render_and_save(os.path.join(save_dir, "final.png"), seed=9999)
except Exception as e:
    print(f"Warning: Failed to save final image: {e}")

plot_loss_history(
    loss_history,
    title=f"{args.optimizer.upper()} Loss Curve",
    save_path=os.path.join(save_dir, "loss_plot.png"),
)

# 動画作成（ログ抑制）
from subprocess import DEVNULL, call

try:
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
except Exception as e:
    print(f"Warning: Failed to create video: {e}")
    print(f"Results saved to {save_dir}/")