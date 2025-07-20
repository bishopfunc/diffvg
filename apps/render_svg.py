"""
SVGからPNGへの変換ユーティリティ

このスクリプトはSVGファイルを読み込み、PNGファイルとして描画する
シンプルなユーティリティです。微分可能レンダリングエンジンを使用して
高品質な画像を生成します。

使用例:
    python render_svg.py input.svg output.png
"""

import os
import argparse
import pydiffvg
import torch as th


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
    
    # シーンをシリアライズして描画準備
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    # 画像を描画
    img = _render(canvas_width, # 幅
                 canvas_height, # 高さ
                 2,   # x方向サンプル数
                 2,   # y方向サンプル数
                 0,   # 乱数シード
                 None,
                 *scene_args)
    return img


def main(args):
    """
    SVG描画のメイン処理を実行します。
    
    引数:
        args: コマンドライン引数
    """
    # GPU デバイスを設定
    pydiffvg.set_device(th.device('cuda:1'))

    # SVGファイルを読み込み
    svg = os.path.join(args.svg)
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(svg)

    # SVGを描画してPNGファイルとして保存
    ref = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(ref.cpu(), args.out, gamma=2.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVGファイルをPNG画像として描画")
    parser.add_argument("svg", help="入力SVGファイルのパス")
    parser.add_argument("out", help="出力画像ファイルのパス")
    args = parser.parse_args()
    main(args)
