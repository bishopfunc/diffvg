"""
ガウシアンブラー効果の最適化デモ

このスクリプトはSVGファイルを読み込み、ガウシアンブラーを適用した目標画像に向けて
ベクターグラフィックスのパラメータ（制御点、線幅、色）を最適化します。
元の鮮明な画像からぼかした画像を再現することで、
微分可能レンダリングの能力を実証します。
"""

import os
import pydiffvg
import torch as th
import scipy.ndimage.filters as F


def render(canvas_width, canvas_height, shapes, shape_groups):
    """
    シェイプとシェイプグループを描画して画像を生成します。
    
    引数:
        canvas_width: キャンバスの幅
        canvas_height: キャンバスの高さ
        shapes: 描画するシェイプのリスト
        shape_groups: シェイプグループのリスト
        
    戻り値:
        描画された画像テンソル
    """
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # 幅
                 canvas_height, # 高さ
                 2,   # x方向サンプル数
                 2,   # y方向サンプル数
                 0,   # 乱数シード
                 None,
                 *scene_args)
    return img


def main():
    """
    ガウシアンブラー最適化のメイン処理を実行します。
    """
    # GPU デバイスを設定
    pydiffvg.set_device(th.device('cuda:1'))

    # SVGファイルを読み込み
    svg = os.path.join("imgs", "peppers.svg")
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(svg)

    # 初期状態を保存
    ref = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(ref.cpu(), 'results/gaussian_blur/init.png', gamma=2.2)

    # ガウシアンフィルターを適用して目標画像を作成
    target = F.gaussian_filter(ref.cpu().numpy(), [10, 10, 0])
    target = th.from_numpy(target).to(ref.device)
    pydiffvg.imwrite(target.cpu(), 'results/gaussian_blur/target.png', gamma=2.2)

    # 最適化する変数を収集
    points_vars = []  # 制御点
    width_vars = []   # 線幅
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        width_vars.append(path.stroke_width)
    
    color_vars = []  # 色（アルファチャンネルは除く）
    for group in shape_groups:
        # アルファチャンネルは最適化しない
        group.fill_color[..., :3].requires_grad = True
        color_vars.append(group.fill_color)

    # オプティマイザーを設定
    points_optim = th.optim.Adam(points_vars, lr=1.0)   # 制御点用
    width_optim = th.optim.Adam(width_vars, lr=1.0)     # 線幅用
    color_optim = th.optim.Adam(color_vars, lr=0.01)    # 色用

    # 最適化ループ（20回反復）
    for t in range(20):
        print('\niteration:', t)
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        
        # フォワードパス: 画像を描画
        img = render(canvas_width, canvas_height, shapes, shape_groups)
        
        # 中間結果を保存
        pydiffvg.imwrite(img.cpu(), 'results/gaussian_blur/iter_{}.png'.format(t), gamma=2.2)
        
        # 損失関数を計算（RGBチャンネルのみ、アルファは除く）
        loss = (img - target)[..., :3].pow(2).mean()

        print('alpha:', img[..., 3].mean().item())
        print('render loss:', loss.item())
    
        # 勾配を逆伝播
        loss.backward()
    
        # 勾配降下ステップを実行
        points_optim.step()
        width_optim.step()
        color_optim.step()
        
        # 色の値を[0, 1]の範囲にクランプ
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

    # 最終結果を描画
    img = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(img.cpu(), 'results/gaussian_blur/final.png', gamma=2.2)

    # 中間描画結果を動画に変換
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/gaussian_blur/iter_%d.png", "-vb", "20M",
        "results/gaussian_blur/out.mp4"])

if __name__ == "__main__":
    main()
