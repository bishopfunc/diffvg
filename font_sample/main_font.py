import pathlib
import random
from pathlib import Path
from subprocess import call
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import ImageReward as RM
import numpy as np
import skimage.io
import torch
from fontTools.misc.transform import Transform
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttCollection import TTCollection
from torchvision import transforms

import pydiffvg

class ImageReward:
    def __init__(self) -> None:
        self.model = RM.load("ImageReward-v1.0")

    def inference_rank(
        self, prompt: str, img_path_list: List[Path]
    ) -> Tuple[List[int], List[float], List[Path]]:
        """画像のランキングとスコアを計算"""
        with torch.no_grad():
            # 画像パスを文字列リストに変換
            img_str_path_list = [str(img_path) for img_path in img_path_list]
        ranking, rewards = self.model.inference_rank(prompt, img_str_path_list)
        return ranking, rewards, img_path_list


def load_target_image(image_path: pathlib.Path, size=(256, 256)) -> torch.Tensor:
    """目標画像を読み込んで Tensor に変換し、リサイズ"""
    image = skimage.io.imread(image_path)
    image = torch.from_numpy(image).float() / 255.0  # [0,1]
    image = image.permute(2, 0, 1)  # [HWC] -> [CHW]
    image = transforms.functional.resize(image, size, antialias=True)
    return image.permute(1, 2, 0)  # [CHW] -> [HWC]


def prepare_initial_path(
    character: str, font_path: pathlib.Path, width: int, height: int
):
    """フォントから文字をSVGパス化し、DiffVGのPath/ShapeGroupとして返す"""
    assert len(character) == 1

    # TTC対応
    if font_path.suffix == ".ttc":
        font = TTCollection(font_path)[0]
    else:
        font = TTFont(font_path)

    glyph_name = font.getBestCmap()[ord(character)]
    glyph_set = font.getGlyphSet()
    pen = SVGPathPen(glyph_set)

    asc, desc = font["OS/2"].sTypoAscender, font["OS/2"].sTypoDescender
    t = Transform(
        width / (asc - desc), 0, 0, -height / (asc - desc), 0, height
    ).translate(y=-desc)
    tpen = TransformPen(pen, t)
    glyph_set[glyph_name].draw(tpen)

    # SVG生成とDiffVG読み込み
    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f'<path d="{pen.getCommands()}"/></svg>'
    )
    with NamedTemporaryFile("w") as tmpf:
        tmpf.write(svg_content)
        tmpf.flush()
        _, _, shapes, shape_groups = pydiffvg.svg_to_scene(tmpf.name)
    path = shapes[0]
    path_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int64),  # ✅ 修正
        fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        stroke_color=None,
    )
    return path, path_group


def optimize_font(
    target: torch.Tensor,
    path,
    path_group,
    steps=100,
    lr=0.02,
    output_dir="results/font_sample",
):
    """DiffVGベースのフォント最適化ループ"""
    canvas_width, canvas_height = target.shape[1], target.shape[0]
    scale = torch.tensor([[canvas_width, canvas_height]], dtype=torch.float32)

    # 初期制御点 + 色を最適化対象として登録
    points_n = path.points.clone().detach()
    points_n += random.uniform(-0.1, 0.1) * torch.rand_like(points_n)
    points_n = torch.nn.Parameter((points_n / scale).detach())
    color = torch.nn.Parameter(torch.tensor([0.3, 0.2, 0.5, 1.0]))

    optimizer = torch.optim.Adam([points_n, color], lr=lr)
    render = pydiffvg.RenderFunction.apply

    best_loss = float("inf")
    best_img = None

    reward_model = ImageReward()
    reward_interval = 3  # 10ステップごとに報酬を計算

    reward = 0.0
    img_path_list = []
    for t in range(steps):
        optimizer.zero_grad()
        path.points = points_n * scale
        path_group.fill_color = color
        shapes, shape_groups = [path], [path_group]

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = render(canvas_width, canvas_height, 2, 2, t + 1, None, *scene_args)
        diffvg_loss = (img - target.contiguous()).pow(2).mean()

        img_path = pathlib.Path(f"{output_dir}/iter_{t:02d}.png")
        pydiffvg.imwrite(img.cpu(), img_path, gamma=2.2)
        img_path_list.append(img_path)

        if (t + 1) % reward_interval == 0:
            latest_img_path_list = img_path_list[-reward_interval:]
            ranking, rewards, _ = reward_model.inference_rank(
                "フォントの美しさを評価してください", latest_img_path_list
            )
            for i, img_path in enumerate(latest_img_path_list):
                print(
                    f"Step {t}: Ranking = {ranking[i]}, Reward = {rewards[i]:.4f}, Image = {latest_img_path_list[i]}"
                )
            reward = np.mean(rewards) # N回でランキングし、平均報酬を計算

        loss = diffvg_loss - reward # 報酬を損失に加える
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_img = img.detach().clone()

        print(
            f"[{t:03d}] loss = {loss.item():.6f}, diffvg_loss = {diffvg_loss.item():.6f}, reward = {reward:.4f}"
        )

        loss.backward()
        optimizer.step()

    # ベスト結果保存
    if best_img is not None:
        pydiffvg.imwrite(best_img.cpu(), f"{output_dir}/best.png", gamma=2.2)

    # 動画出力
    call(
        [
            "ffmpeg",
            "-framerate",
            "24",
            "-i",
            f"{output_dir}/iter_%02d.png",
            "-vb",
            "20M",
            f"{output_dir}/out.mp4",
        ]
    )
    print(f"Saved optimization video: {output_dir}/out.mp4")


def main():
    # pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_use_gpu(False)  # GPUを使わない設定

    canvas_size = (256, 256)
    character = "あ"
    font_path = pathlib.Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc")
    target_path = pathlib.Path("results/font_sample/target.png")
    output_dir = "results/font_sample"

    target = load_target_image(target_path, size=canvas_size)
    path, path_group = prepare_initial_path(character, font_path, *canvas_size)
    optimize_font(target, path, path_group, steps=100, lr=0.02, output_dir=output_dir)


if __name__ == "__main__":
    main()
