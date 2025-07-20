"""
シームカービングによるSVGリターゲティング

このスクリプトは画像領域でのシームカービングを使用してSVGを縮小します。
エネルギー関数に基づいて重要度の低い縦方向のシームを除去し、
微分可能レンダリングを使用してベクター形状を最適化します。
"""

import os
import pydiffvg
import argparse
import torch as th
import scipy.ndimage.filters as filters
import numba
import numpy as np
import skimage.io


def energy(im):
    """
    画像のエネルギーを計算します。
    
    Sobelフィルターを使用して勾配を計算し、
    エッジの強度をエネルギーとして使用します。

    引数:
        im(np.ndarray) 形状[h, w, 3]: 入力画像

    戻り値:
        (np.ndarray) 形状[h, w]: エネルギーマップ
    """
    # Sobelフィルター（x方向）
    f_dx = np.array([
        [-1, 0, 1 ],
        [-2, 0, 2 ],
        [-1, 0, 1 ],
    ])
    # Sobelフィルター（y方向）
    f_dy = f_dx.T
    
    # グレースケール画像に対して勾配を計算
    dx = filters.convolve(im.mean(2), f_dx)
    dy = filters.convolve(im.mean(2), f_dy)

    # 勾配の絶対値の和をエネルギーとする
    return np.abs(dx) + np.abs(dy)


@numba.jit(nopython=True)
def min_seam(e):
    """
    エネルギーマップから最小コストのシームを見つけます。
    
    動的プログラミングを使用して、上から下への最小エネルギーパスを計算します。

    引数:
        e(np.ndarray) 形状[h, w]: エネルギーマップ
    
    戻り値:
        min_e(np.ndarray) 形状[h, w]: 各位置(y,x)について、
            上端から位置yまでの最小シームコスト
        argmin_e(np.ndarray) 形状[h, w]: 各位置(y,x)について、
            前の行(y-1)での最適パスのx座標（バックトラック用）
    """
    # ローカルエネルギーで初期化
    min_e = e.copy()
    argmin_e = np.zeros_like(e, dtype=np.int64)

    h, w = e.shape

    # 縦方向に伝播
    for y in range(1, h):
        for x in range(w):
            # 左端の場合
            if x == 0:
                idx = np.argmin(e[y-1, x:x+2])
                argmin_e[y, x] = idx + x
                mini = e[y-1, x + idx]
            # 右端の場合
            elif x == w-1:
                idx = np.argmin(e[y-1, x-1:x+1])
                argmin_e[y, x] = idx + x - 1
                mini = e[y-1, x + idx - 1]
            # 中央の場合
            else:
                idx = np.argmin(e[y-1, x-1:x+2])
                argmin_e[y, x] = idx + x - 1
                mini = e[y-1, x + idx - 1]

            # 累積最小コストを更新
            min_e[y, x] = min_e[y, x] + mini

    return min_e, argmin_e


def carve_seam(im):
    """
    画像から縦方向のシームを除去し、横幅を1ピクセル縮小します。

    引数:
        im(np.ndarray) 形状[h, w, 3]: 入力画像

    戻り値:
        (np.ndarray) 形状[h, w-1, 3]: シームが除去された画像
    """
    # エネルギーマップを計算
    e = energy(im)
    min_e, argmin_e = min_seam(e)
    h, w = im.shape[:2]

    # 保持するピクセルのブール値フラグ
    to_keep = np.ones((h, w), dtype=np.bool)

    # 最下行から最小エネルギーの位置を取得
    x = np.argmin(min_e[-1])
    print("carving seam", x, "with energy", min_e[-1, x])

    # バックトラックしてシームを特定
    for y in range(h-1, -1, -1):
        # シームピクセルを除去
        to_keep[y, x] = False
        x = argmin_e[y, x]

    # マスクをカラーチャンネルに複製
    to_keep = np.stack(3*[to_keep], axis=2)
    new_im = im[to_keep].reshape((h, w-1, 3))
    return new_im


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    """
    シェイプとシェイプグループを描画して画像を生成します。
    
    引数:
        canvas_width: キャンバスの幅
        canvas_height: キャンバスの高さ
        shapes: 描画するシェイプのリスト
        shape_groups: シェイプグループのリスト
        samples: サンプリング数（デフォルト: 2）
        
    戻り値:
        描画された画像テンソル
    """
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    img = _render(canvas_width, # 幅
                 canvas_height, # 高さ
                 samples,   # x方向サンプル数
                 samples,   # y方向サンプル数
                 0,   # 乱数シード
                 None,
                 *scene_args)
    return img


def vector_rescale(shapes, scale_x=1.00, scale_y=1.00):
    """
    ベクター形状の座標をスケーリングします。
    
    引数:
        shapes: スケーリングするシェイプのリスト
        scale_x: x方向のスケール係数
        scale_y: y方向のスケール係数
    """
    for path in shapes:
        path.points[..., 0] *= scale_x
        path.points[..., 1] *= scale_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", default=os.path.join("imgs", "hokusai.svg"))
    parser.add_argument("--optim_steps", default=10, type=int)
    parser.add_argument("--lr", default=1e-1, type=int)
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.svg))[0]
    root = os.path.join("results", "seam_carving", name)
    svg_root = os.path.join(root, "svg")
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "svg"), exist_ok=True)

    pydiffvg.set_use_gpu(False)
    # pydiffvg.set_device(th.device('cuda'))

    # Load SVG
    print("loading svg %s" % args.svg)
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    print("done loading")

    max_size = 512
    scale_factor = max_size / max(canvas_width, canvas_height)
    print("rescaling from %dx%d with scale %f" % (canvas_width, canvas_height, scale_factor))
    canvas_width = int(canvas_width*scale_factor)
    canvas_height = int(canvas_height*scale_factor)
    print("new shape %dx%d" % (canvas_width, canvas_height))
    vector_rescale(shapes, scale_x=scale_factor, scale_y=scale_factor)

    # Shrink image by 33 %
    # num_seams_to_remove = 2
    num_seams_to_remove = canvas_width // 3
    new_canvas_width  = canvas_width - num_seams_to_remove
    scaling =  new_canvas_width * 1.0 / canvas_width

    # Naive scaling baseline
    print("rendering naive rescaling...")
    vector_rescale(shapes, scale_x=scaling)
    resized = render(new_canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(resized.cpu(), os.path.join(root, 'uniform_scaling.png'), gamma=2.2)
    pydiffvg.save_svg(os.path.join(svg_root, 'uniform_scaling.svg') , canvas_width,
                      canvas_height, shapes, shape_groups, use_gamma=False)
    vector_rescale(shapes, scale_x=1.0/scaling)  # bring back original coordinates
    print("saved naiving scaling")

    # Save initial state
    print("rendering initial state...")
    im = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(im.cpu(), os.path.join(root, 'init.png'), gamma=2.2)
    pydiffvg.save_svg(os.path.join(svg_root, 'init.svg'), canvas_width,
                      canvas_height, shapes, shape_groups, use_gamma=False)
    print("saved initial state")

    # Optimize
    # color_optim = th.optim.Adam(color_vars, lr=0.01)

    retargeted = im[..., :3].cpu().numpy()
    previous_width = canvas_width
    print("carving seams")
    for seam_idx in range(num_seams_to_remove):
        print('\nseam', seam_idx+1, 'of', num_seams_to_remove)

        # Remove a seam
        retargeted = carve_seam(retargeted)

        current_width = canvas_width - seam_idx - 1
        scale_factor = current_width * 1.0 / previous_width
        previous_width = current_width

        padded = np.zeros((canvas_height, canvas_width, 4))
        padded[:, :-seam_idx-1, :3] = retargeted
        padded[:, :-seam_idx-1, -1] = 1.0  # alpha
        padded = th.from_numpy(padded).to(im.device)

        # Remap points to the smaller canvas and
        # collect variables to optimize
        points_vars = []
        # width_vars = []
        mini, maxi = canvas_width, 0
        for path in shapes:
            path.points.requires_grad = False
            x = path.points[..., 0]
            y = path.points[..., 1]
            # rescale

            x = x * scale_factor

            # clip to canvas
            path.points[..., 0] = th.clamp(x, 0, current_width)
            path.points[..., 1] = th.clamp(y, 0, canvas_height)

            path.points.requires_grad = True
            points_vars.append(path.points)
            path.stroke_width.requires_grad = True
            # width_vars.append(path.stroke_width)

            mini = min(mini, path.points.min().item())
            maxi = max(maxi, path.points.max().item())
        print("points", mini, maxi, "scale", scale_factor)

        # recreate an optimizer so we don't carry over the previous update
        # (momentum)?
        geom_optim = th.optim.Adam(points_vars, lr=args.lr)

        for step in range(args.optim_steps):
            geom_optim.zero_grad()

            img = render(canvas_width, canvas_height, shapes, shape_groups,
                         samples=2)

            pydiffvg.imwrite(
                img.cpu(), 
                os.path.join(root, "seam_%03d_iter_%02d.png" % (seam_idx, step)), gamma=2.2)

            # NO alpha
            loss = (img - padded)[..., :3].pow(2).mean()
            # loss = (img - padded).pow(2).mean()
            print('render loss:', loss.item())

            # Backpropagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            geom_optim.step()
        pydiffvg.save_svg(os.path.join(svg_root, "seam%03d.svg" % seam_idx),
                          canvas_width-seam_idx, canvas_height, shapes,
                          shape_groups, use_gamma=False)

        for path in shapes:
            mini = min(mini, path.points.min().item())
            maxi = max(maxi, path.points.max().item())
        print("points", mini, maxi)

    img = render(canvas_width, canvas_height, shapes, shape_groups)
    img = img[:, :-num_seams_to_remove]

    pydiffvg.imwrite(img.cpu(), os.path.join(root, 'final.png'),
                     gamma=2.2)
    pydiffvg.imwrite(retargeted, os.path.join(root, 'ref.png'),
                     gamma=2.2)

    pydiffvg.save_svg(os.path.join(svg_root, 'final.svg'),
                      canvas_width-seam_idx, canvas_height, shapes,
                      shape_groups, use_gamma=False)

    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i", os.path.join(root, "seam_%03d_iter_00.png"), "-vb", "20M",
         os.path.join(root, "out.mp4")])


if __name__ == "__main__":
    main()
