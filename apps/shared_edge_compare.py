"""
共有エッジ比較ツール

このスクリプトは共有エッジを持つSVG形状の勾配計算を比較し、
前進差分、後退差分、および解析的勾配の違いを可視化します。
特定の形状（shapes[2]）のみを摂動させて、
エッジ共有による勾配への影響を調査します。
"""

import pydiffvg
import diffvg
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
import torch

def normalize(x, min_, max_):
    """
    値を[0, 1]の範囲に正規化します。
    
    引数:
        x: 正規化する値
        min_: 最小値
        max_: 最大値
        
    戻り値:
        正規化された値
    """
    range = max(abs(min_), abs(max_))
    return (x + range) / (2 * range)

def main(args):
    """
    共有エッジ比較のメイン処理を実行します。
    
    引数:
        args: コマンドライン引数
    """
    # SVGファイルを読み込んでシーンを構築
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_file)

    # スケールを適用した画像サイズを計算
    w = int(canvas_width * args.size_scale)
    h = int(canvas_height * args.size_scale)

    # ボックスフィルターを設定
    pfilter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                   radius = torch.tensor(0.5))

    # プリフィルタリングを無効化
    use_prefiltering = False
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)

    # 高サンプリング数で元画像を描画
    num_samples_x = 16
    num_samples_y = 16
    render = pydiffvg.RenderFunction.apply
    img = render(w, # 幅
                 h, # 高さ
                 num_samples_x, # x方向サンプル数
                 num_samples_y, # y方向サンプル数
                 0, # 乱数シード
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/finite_difference_comp/img.png', gamma=1.0)

    # 有限差分法のための摂動量
    epsilon = 0.1
    
    def perturb_scene(axis, epsilon):
        """
        特定の形状（shapes[2]）のみを指定軸方向に摂動させます。
        
        引数:
            axis: 摂動する軸（0=x軸, 1=y軸）
            epsilon: 摂動量
        """
        # 3番目の形状のみを摂動（共有エッジの影響を調査）
        shapes[2].points[:, axis] += epsilon
        # 以下はコメントアウト：全ての形状を摂動する場合のコード
        # for s in shapes:
        #     if isinstance(s, pydiffvg.Circle):
        #         s.center[axis] += epsilon
        #     elif isinstance(s, pydiffvg.Ellipse):
        #         s.center[axis] += epsilon
        #     elif isinstance(s, pydiffvg.Path):
        #         s.points[:, axis] += epsilon
        #     elif isinstance(s, pydiffvg.Polygon):
        #         s.points[:, axis] += epsilon
        #     elif isinstance(s, pydiffvg.Rect):
        #         s.p_min[axis] += epsilon
        #         s.p_max[axis] += epsilon
        # for s in shape_groups:
        #     if isinstance(s.fill_color, pydiffvg.LinearGradient):
        #         s.fill_color.begin[axis] += epsilon
        #         s.fill_color.end[axis] += epsilon

    # === 前進差分の計算 ===
    
    # x方向に+epsilon摂動して描画
    perturb_scene(0, epsilon)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    render = pydiffvg.RenderFunction.apply
    img0 = render(w, # 幅
                  h, # 高さ
                  num_samples_x,   # x方向サンプル数
                  num_samples_y,   # y方向サンプル数
                  0,   # 乱数シード
                  None,
                  *scene_args)

    # 前進差分を計算
    forward_diff = (img0 - img) / (epsilon)
    forward_diff = forward_diff.sum(axis = 2)  # RGBAチャンネルを合計
    x_diff_max = 1.5
    x_diff_min = -1.5
    print(forward_diff.max())
    print(forward_diff.min())
    
    # カラーマップを適用して保存
    forward_diff = cm.viridis(normalize(forward_diff, x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(forward_diff, 'results/finite_difference_comp/shared_edge_forward_diff.png', gamma=1.0)

    # === 後退差分の計算 ===
    
    # x方向に-2*epsilon摂動して描画（合計-epsilon移動）
    perturb_scene(0, -2 * epsilon)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    img1 = render(w, # 幅
                  h, # 高さ
                  num_samples_x,   # x方向サンプル数
                  num_samples_y,   # y方向サンプル数
                  0,   # 乱数シード
                  None,
                  *scene_args)
    
    # 後退差分を計算
    backward_diff = (img - img1) / (epsilon)
    backward_diff = backward_diff.sum(axis = 2)  # RGBAチャンネルを合計
    print(backward_diff.max())
    print(backward_diff.min())
    
    # カラーマップを適用して保存
    backward_diff = cm.viridis(normalize(backward_diff, x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(backward_diff, 'results/finite_difference_comp/shared_edge_backward_diff.png', gamma=1.0)
    
    # 摂動を元に戻す
    perturb_scene(0, epsilon)

    # === 解析的勾配の計算 ===
    
    # サンプリング数を下げて解析的勾配を計算
    num_samples_x = 4
    num_samples_y = 4
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    render_grad = pydiffvg.RenderFunction.render_grad
    img_grad = render_grad(torch.ones(h, w, 4),
                           w, # 幅
                           h, # 高さ
                           num_samples_x, # x方向サンプル数
                           num_samples_y, # y方向サンプル数
                           0, # 乱数シード
                           *scene_args)
    print(img_grad[:, :, 0].max())
    print(img_grad[:, :, 0].min())
    
    # 解析的勾配を同じスケールで可視化
    x_diff = cm.viridis(normalize(img_grad[:, :, 0], x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(x_diff, 'results/finite_difference_comp/ours_x_diff.png', gamma=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="共有エッジを持つ形状の勾配比較")
    parser.add_argument("svg_file", help="入力SVGファイルのパス")
    parser.add_argument("--size_scale", type=float, default=1.0,
                       help="画像サイズのスケール係数 (デフォルト: 1.0)")
    args = parser.parse_args()
    main(args)
