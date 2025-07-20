"""
有限差分法と解析的勾配の比較を行うモジュール

このスクリプトはSVGファイルを読み込み、有限差分法で計算した勾配と
解析的に計算した勾配を比較し、視覚化します。

使用例:
    python finite_difference_comp.py imgs/tiger.svg 
    python finite_difference_comp.py --use_prefiltering True imgs/tiger.svg 
    python finite_difference_comp.py imgs/boston.svg
    python finite_difference_comp.py --use_prefiltering True imgs/boston.svg
    python finite_difference_comp.py imgs/contour.svg
    python finite_difference_comp.py --use_prefiltering True imgs/contour.svg
    python finite_difference_comp.py --size_scale 0.5 --clamping_factor 0.05 imgs/hawaii.svg
    python finite_difference_comp.py --size_scale 0.5 --clamping_factor 0.05 --use_prefiltering True imgs/hawaii.svg
    python finite_difference_comp.py imgs/mcseem2.svg
    python finite_difference_comp.py --use_prefiltering True imgs/mcseem2.svg
    python finite_difference_comp.py imgs/reschart.svg
    python finite_difference_comp.py --use_prefiltering True imgs/reschart.svg
"""

import pydiffvg
import diffvg
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
import torch

# タイミング情報を表示
pydiffvg.set_print_timing(True)
# GPU使用を無効化（必要に応じてコメントアウト）
#pydiffvg.set_use_gpu(False)

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
    有限差分法と解析的勾配の比較を実行するメイン関数。
    
    引数:
        args: コマンドライン引数
    """
    # SVGファイルを読み込んでシーンを構築
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg_file)

    # スケールを適用した画像サイズを計算
    w = int(canvas_width * args.size_scale)
    h = int(canvas_height * args.size_scale)

    print(w, h)
    
    # シーン内の曲線数をカウント
    curve_counts = 0
    for s in shapes:
        if isinstance(s, pydiffvg.Circle):
            curve_counts += 1
        elif isinstance(s, pydiffvg.Ellipse):
            curve_counts += 1
        elif isinstance(s, pydiffvg.Path):
            curve_counts += len(s.num_control_points)
        elif isinstance(s, pydiffvg.Polygon):
            curve_counts += len(s.points) - 1
            if s.is_closed:
                curve_counts += 1
        elif isinstance(s, pydiffvg.Rect):
            curve_counts += 1
    print('curve_counts:', curve_counts)

    # ピクセルフィルターを設定（ボックスフィルター、半径0.5）
    pfilter = pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                   radius = torch.tensor(0.5))

    use_prefiltering = args.use_prefiltering
    print('use_prefiltering:', use_prefiltering)

    # シーンをシリアライズして描画準備
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)

    # サンプリング数を設定
    num_samples_x = args.num_spp
    num_samples_y = args.num_spp
    if (use_prefiltering):
        # プリフィルタリング使用時はサンプリング数を1に
        num_samples_x = 1
        num_samples_y = 1

    # 元画像を描画
    render = pydiffvg.RenderFunction.apply
    img = render(w, # 幅
                 h, # 高さ
                 num_samples_x, # x方向サンプル数
                 num_samples_y, # y方向サンプル数
                 0, # 乱数シード
                 None, # 背景画像
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/finite_difference_comp/img.png', gamma=1.0)

    # 有限差分法のための摂動量
    epsilon = 0.1
    
    def perturb_scene(axis, epsilon):
        """
        シーン内の全ての形状を指定軸方向に摂動させます。
        
        引数:
            axis: 摂動する軸（0=x軸, 1=y軸）
            epsilon: 摂動量
        """
        # 全ての形状を摂動
        for s in shapes:
            if isinstance(s, pydiffvg.Circle):
                s.center[axis] += epsilon
            elif isinstance(s, pydiffvg.Ellipse):
                s.center[axis] += epsilon
            elif isinstance(s, pydiffvg.Path):
                s.points[:, axis] += epsilon
            elif isinstance(s, pydiffvg.Polygon):
                s.points[:, axis] += epsilon
            elif isinstance(s, pydiffvg.Rect):
                s.p_min[axis] += epsilon
                s.p_max[axis] += epsilon
        
        # グラデーションも摂動
        for s in shape_groups:
            if isinstance(s.fill_color, pydiffvg.LinearGradient):
                s.fill_color.begin[axis] += epsilon
                s.fill_color.end[axis] += epsilon

    # === X方向の有限差分勾配計算 ===
    
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
                  None, # 背景画像
                  *scene_args)

    # x方向に-epsilon摂動して描画（合計-2*epsilon移動）
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
                  None, # 背景画像
                  *scene_args)
    
    # x方向の有限差分勾配を計算
    x_diff = (img0 - img1) / (2 * epsilon)
    x_diff = x_diff.sum(axis = 2)  # RGBAチャンネルを合計
    x_diff_max = x_diff.max() * args.clamping_factor
    x_diff_min = x_diff.min() * args.clamping_factor
    print(x_diff.max())
    print(x_diff.min())
    
    # カラーマップを適用して保存
    x_diff = cm.viridis(normalize(x_diff, x_diff_min, x_diff_max).cpu().numpy())
    pydiffvg.imwrite(x_diff, 'results/finite_difference_comp/finite_x_diff.png', gamma=1.0)

    # x方向の摂動を元に戻す
    perturb_scene(0, epsilon)

    # === Y方向の有限差分勾配計算 ===
    
    # y方向に+epsilon摂動して描画
    perturb_scene(1, epsilon)
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
                  None, # 背景画像
                  *scene_args)

    # y方向に-epsilon摂動して描画（合計-2*epsilon移動）
    perturb_scene(1, -2 * epsilon)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    img1 = render(w, # 幅
                  h, # 高さ
                  num_samples_x,   # x方向サンプル数
                  num_samples_y,   # y方向サンプル数
                  0,   # 乱数シード
                  None, # 背景画像
                  *scene_args)
    
    # y方向の有限差分勾配を計算
    y_diff = (img0 - img1) / (2 * epsilon)
    y_diff = y_diff.sum(axis = 2)  # RGBAチャンネルを合計
    y_diff_max = y_diff.max() * args.clamping_factor
    y_diff_min = y_diff.min() * args.clamping_factor
    
    # カラーマップを適用して保存
    y_diff = cm.viridis(normalize(y_diff, y_diff_min, y_diff_max).cpu().numpy())
    pydiffvg.imwrite(y_diff, 'results/finite_difference_comp/finite_y_diff.png', gamma=1.0)
    
    # y方向の摂動を元に戻す
    perturb_scene(1, epsilon)

    # === 解析的勾配の計算 ===
    
    # シーンを再構築
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups,
        filter = pfilter,
        use_prefiltering = use_prefiltering)
    
    # 解析的勾配を計算
    render_grad = pydiffvg.RenderFunction.render_grad
    img_grad = render_grad(torch.ones(h, w, 4, device = pydiffvg.get_device()),
                           w, # 幅
                           h, # 高さ
                           num_samples_x, # x方向サンプル数
                           num_samples_y, # y方向サンプル数
                           0, # 乱数シード
                           None, # 背景画像
                           *scene_args)
    
    print(img_grad[:, :, 0].max())
    print(img_grad[:, :, 0].min())
    
    # 解析的勾配を同じスケールで可視化
    x_diff = cm.viridis(normalize(img_grad[:, :, 0], x_diff_min, x_diff_max).cpu().numpy())
    y_diff = cm.viridis(normalize(img_grad[:, :, 1], y_diff_min, y_diff_max).cpu().numpy())
    pydiffvg.imwrite(x_diff, 'results/finite_difference_comp/ours_x_diff.png', gamma=1.0)
    pydiffvg.imwrite(y_diff, 'results/finite_difference_comp/ours_y_diff.png', gamma=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="有限差分法と解析的勾配の比較")
    parser.add_argument("svg_file", help="入力SVGファイルのパス")
    parser.add_argument("--size_scale", type=float, default=1.0, 
                       help="画像サイズのスケール係数 (デフォルト: 1.0)")
    parser.add_argument("--clamping_factor", type=float, default=0.1,
                       help="勾配値のクランプ係数 (デフォルト: 0.1)")
    parser.add_argument("--num_spp", type=int, default=4,
                       help="ピクセルあたりのサンプル数 (デフォルト: 4)")
    parser.add_argument("--use_prefiltering", type=bool, default=False,
                       help="プリフィルタリングを使用するかどうか (デフォルト: False)")
    args = parser.parse_args()
    main(args)
