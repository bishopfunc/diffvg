"""
SVG最適化ツール

このスクリプトは既存のSVGファイルを読み込み、目標画像に近づけるように
制御点と色を最適化します。勾配降下法を使用して、
SVGの表現力を活用した画像近似を行います。

使用例:
    python refine_svg.py input.svg target.png --num_iter 500
    python refine_svg.py input.svg target.png --use_lpips_loss --num_iter 250
"""

import pydiffvg
import argparse
import ttools.modules
import torch
import skimage.io

# ガンマ補正値
gamma = 1.0

def main(args):
    """
    SVG最適化のメイン処理を実行します。
    
    引数:
        args: コマンドライン引数
    """
    # LPIPS知覚損失関数を初期化
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    # 目標画像を読み込み、前処理を実行
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)  # ガンマ補正を適用
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)  # バッチ次元を追加
    target = target.permute(0, 3, 1, 2)  # NHWC -> NCHW形式に変換

    # SVGファイルを読み込んでシーンを構築
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    
    # 初期シーンをシリアライズ
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    # 初期画像を描画
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # 幅
                 canvas_height, # 高さ
                 2,   # x方向サンプル数
                 2,   # y方向サンプル数
                 0,   # 乱数シード
                 None, # 背景画像
                 *scene_args)
    
    # 出力画像はリニアRGB空間なので、保存前にガンマ補正を適用
    pydiffvg.imwrite(img.cpu(), 'results/refine_svg/init.png', gamma=gamma)

    # 最適化する変数を収集
    points_vars = []  # 制御点
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    
    # 色変数（重複を除去）
    color_vars = {}
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars[group.fill_color.data_ptr()] = group.fill_color
    color_vars = list(color_vars.values())

    # オプティマイザーを設定
    points_optim = torch.optim.Adam(points_vars, lr=1.0)   # 制御点用
    color_optim = torch.optim.Adam(color_vars, lr=0.01)    # 色用

    # 最適化ループ
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()
        
        # フォワードパス: 画像を描画
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # 幅
                     canvas_height, # 高さ
                     2,   # x方向サンプル数
                     2,   # y方向サンプル数
                     0,   # 乱数シード
                     None, # 背景画像
                     *scene_args)
        
        # 白背景と合成（アルファブレンディング）
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        
        # 中間結果を保存
        pydiffvg.imwrite(img.cpu(), 'results/refine_svg/iter_{}.png'.format(t), gamma=gamma)
        
        # RGBチャンネルのみを使用
        img = img[:, :, :3]
        
        # HWC -> NCHW形式に変換
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        
        # 損失関数を計算
        if args.use_lpips_loss:
            loss = perception_loss(img, target)  # LPIPS知覚損失
        else:
            loss = (img - target).pow(2).mean()  # L2損失
        print('render loss:', loss.item())
    
        # 勾配を逆伝播
        loss.backward()
    
        # 勾配降下ステップを実行
        points_optim.step()
        color_optim.step()
        
        # 色の値を[0, 1]の範囲にクランプ
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        # 定期的にSVGファイルを保存
        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('results/refine_svg/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)

    # 最終結果を描画
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, # 幅
                 canvas_height, # 高さ
                 2,   # x方向サンプル数
                 2,   # y方向サンプル数
                 0,   # 乱数シード
                 None, # 背景画像
                 *scene_args)
    
    # 最終画像を保存
    pydiffvg.imwrite(img.cpu(), 'results/refine_svg/final.png'.format(t), gamma=gamma)
    
    # 中間描画結果を動画に変換
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/refine_svg/iter_%d.png", "-vb", "20M",
        "results/refine_svg/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVGファイルを目標画像に向けて最適化")
    parser.add_argument("svg", help="入力SVGファイルのパス")
    parser.add_argument("target", help="目標画像のパス")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true',
                       help="LPIPS知覚損失を使用するかどうか")
    parser.add_argument("--num_iter", type=int, default=250,
                       help="最適化の反復回数 (デフォルト: 250)")
    args = parser.parse_args()
    main(args)
