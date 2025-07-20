"""
三次ベジエ曲線を二次ベジエ曲線に分割・変換するモジュール

このモジュールは三次ベジエ曲線を適応的に分割し、
各セグメントを二次ベジエ曲線で近似することで、
より単純な曲線表現に変換します。
"""

import svgpathtools
import numpy as np
import math

def split_cubic(c, t):
    """
    三次ベジエ曲線を指定されたパラメータtで2つの三次ベジエ曲線に分割します。
    
    引数:
        c: 分割する三次ベジエ曲線
        t: 分割点のパラメータ（0.0-1.0）
        
    戻り値:
        分割された2つの三次ベジエ曲線のタプル
    """
    c0, c1 = svgpathtools.split_bezier(c, t)
    return svgpathtools.CubicBezier(c0[0], c0[1], c0[2], c0[3]), svgpathtools.CubicBezier(c1[0], c1[1], c1[2], c1[3])

def cubic_to_quadratic(curve):
    """
    三次ベジエ曲線を二次ベジエ曲線に変換します。
    
    最適なL2近似を使用して、三次曲線を二次曲線で近似します。
    制御点は三次曲線の制御点の重み付き平均として計算されます。
    
    引数:
        curve: 変換する三次ベジエ曲線
        
    戻り値:
        近似された二次ベジエ曲線
    """
    # 最適なL2近似による制御点計算
    m = (-curve.start + 3 * curve.control1 + 3 * curve.control2 - curve.end) / 4.0
    return svgpathtools.QuadraticBezier(curve.start, m, curve.end)

def convert_and_write_svg(cubic, filename):
    """
    三次ベジエ曲線を適応的に二次ベジエ曲線に変換し、SVGファイルとして出力します。
    
    この関数は三次ベジエ曲線を分析し、必要に応じて分割しながら
    二次ベジエ曲線で近似し、元の曲線と近似曲線を視覚化したSVGを生成します。
    
    引数:
        cubic: 変換する三次ベジエ曲線
        filename: 出力するSVGファイル名
    """
    # 元の三次ベジエ曲線とその制御線を作成
    cubic_path = svgpathtools.Path(cubic)
    cubic_ctrl = svgpathtools.Path(svgpathtools.Line(cubic.start, cubic.control1),
                                   svgpathtools.Line(cubic.control1, cubic.control2),
                                   svgpathtools.Line(cubic.control2, cubic.end))
    
    # 色設定
    cubic_color = (50, 50, 200)        # 三次曲線: 青色
    cubic_ctrl_color = (150, 150, 150) # 制御線: 灰色

    r = 4.0  # 制御点の半径

    # 描画要素を初期化
    paths = [cubic_path, cubic_ctrl]
    colors = [cubic_color, cubic_ctrl_color]
    dots = [cubic_path[0].start, cubic_path[0].control1, cubic_path[0].control2, cubic_path[0].end]
    ncols = ['green', 'green', 'green', 'green']  # 制御点の色
    nradii = [r, r, r, r]  # 制御点の半径
    stroke_widths = [3.0, 1.5]  # 線の太さ

    def add_quadratic(q):
        """
        二次ベジエ曲線を描画リストに追加します。
        
        引数:
            q: 追加する二次ベジエ曲線
        """
        paths.append(q)
        # 二次曲線の制御線を作成
        q_ctrl = svgpathtools.Path(svgpathtools.Line(q.start, q.control),
                                   svgpathtools.Line(q.control, q.end))
        paths.append(q_ctrl)
        colors.append((200, 50, 50))     # 二次曲線: 赤色
        colors.append((150, 150, 150))   # 制御線: 灰色
        
        # 制御点を追加
        dots.append(q.start)
        dots.append(q.control)
        dots.append(q.end)
        ncols.append('purple')  # 二次曲線の制御点: 紫色
        ncols.append('purple')
        ncols.append('purple')
        nradii.append(r)
        nradii.append(r)
        nradii.append(r)
        stroke_widths.append(3.0)  # 曲線の太さ
        stroke_widths.append(1.5)  # 制御線の太さ

    # 適応的分割のための変数
    prec = 1.0  # 精度パラメータ（未使用）
    queue = [cubic]  # 処理待ちの曲線キュー
    num_quadratics = 0  # 生成された二次曲線の数

    # 適応的分割と変換のメインループ
    while len(queue) > 0:
        c = queue[-1]  # キューから曲線を取得
        queue = queue[:-1]

        # 変換基準の計算
        # 参考: http://caffeineowl.com/graphics/2d/vectorial/cubic2quad01.html
        p = c.end - 3 * c.control2 + 3 * c.control1 - c.start
        d = math.sqrt(p.real * p.real + p.imag * p.imag) * math.sqrt(3.0) / 36
        t = math.pow(1.0 / d, 1.0 / 3.0)

        if t < 1.0:
            # 曲線が複雑すぎる場合は分割
            c0, c1 = split_cubic(c, 0.5)
            queue.append(c0)
            queue.append(c1)
        else:
            # 十分単純な場合は二次曲線に変換
            quadratic = cubic_to_quadratic(c)
            print(quadratic)
            add_quadratic(quadratic)
            num_quadratics += 1
    
    print('num_quadratics:', num_quadratics)

    # SVGファイルとして出力
    svgpathtools.wsvg(paths,
                      colors = colors,
                      stroke_widths = stroke_widths,
                      nodes = dots,
                      node_colors = ncols,
                      node_radii = nradii,
                      filename = filename)

# テスト用の三次ベジエ曲線を定義して変換を実行
# 複素数表記で制御点を定義: 実部=x座標, 虚部=y座標

# テストケース1: 標準的な三次ベジエ曲線
convert_and_write_svg(svgpathtools.CubicBezier(100+200j, 426+50j, 50+50j, 300+200j),
                      'results/curve_subdivision/subdiv_curve0.svg')

# テストケース2: わずかに異なる制御点を持つ曲線（比較用）
convert_and_write_svg(svgpathtools.CubicBezier(100+200j, 427+50j, 50+50j, 300+200j),
                      'results/curve_subdivision/subdiv_curve1.svg')
