# diffvg/apps ディレクトリ概要

このディレクトリには、微分可能ベクターグラフィックス（diffvg）ライブラリを使用した様々なデモンストレーションとアプリケーションが含まれています。

## 基本的な形状最適化

### 単一形状の最適化
- **single_circle.py** - 単一円の最適化デモ（半径、中心、色の最適化）
- **single_circle_outline.py** - アウトライン付き円の最適化（塗りつぶしとストロークの両方）
- **single_circle_sdf.py** - SDF（符号付き距離場）を使用した円の最適化
- **single_circle_tf.py** - TensorFlow版の円最適化デモ
- **single_ellipse.py** - 楕円の最適化（半径、中心、色の最適化）
- **single_ellipse_transform.py** - アフィン変換を使用した楕円の最適化
- **single_rect.py** - 矩形の最適化（位置、サイズ、色の最適化）
- **single_polygon.py** - 多角形の最適化（頂点位置と色の最適化）

### 曲線とパスの最適化
- **single_curve.py** - 単一ベジエ曲線の最適化デモ（制御点と色の最適化）
- **single_curve_outline.py** - アウトライン付きベジエ曲線の最適化
- **single_curve_sdf.py** - SDFを使用したベジエ曲線の最適化
- **single_curve_sdf_trans.py** - SDF変換を使用した曲線最適化の実験的実装
- **single_curve_tf.py** - TensorFlow版のベジエ曲線最適化
- **single_open_curve.py** - 開いた曲線（ストロークのみ）の最適化
- **single_open_curve_thickness.py** - 可変線幅を持つ開いた曲線の最適化
- **single_path.py** - 複雑なSVGパスの最適化
- **single_path_sdf.py** - SDFを使用したSVGパスの最適化
- **single_stroke.py** - ストローク専用パスの最適化
- **single_stroke_tf.py** - TensorFlow版のストローク最適化

### グラデーションと高度な効果
- **single_gradient.py** - 線形グラデーションを使用した形状の最適化

## 画像処理とフィルタリング

### フィルター最適化
- **optimize_pixel_filter.py** - ピクセルフィルター半径パラメータの最適化デモ
- **gaussian_blur.py** - ガウシアンブラー効果の最適化デモ

### 画像比較と解析
- **image_compare.py** - 2つの画像を参照画像と比較し、品質メトリクス（MSE、PSNR、SSIM）を計算
- **finite_difference_comp.py** - 有限差分法と解析的勾配の比較・可視化
- **shared_edge_compare.py** - 共有エッジでの勾配計算の比較（実験的）

## SVG処理とレンダリング

### SVG変換とレンダリング
- **render_svg.py** - SVGファイルをPNG画像として描画するユーティリティ
- **refine_svg.py** - 既存のSVGファイルを目標画像に向けて最適化
- **simple_transform_svg.py** - SVGのアフィン変換パラメータを最適化（Visdom可視化付き）
- **svg_parse_test.py** - SVGパース機能のテストスクリプト

### 高度なSVG最適化
- **painterly_rendering.py** - 入力画像を微分可能ベクターパスで絵画風に近似
- **style_transfer.py** - ニューラルスタイル転送をベクターグラフィックスに適用
- **svg_brush.py** - インタラクティブなSVGブラシツール（Pygame使用）

## 特殊な応用

### 画像リターゲティング
- **seam_carving.py** - シームカービングを使用したSVGの適応的リサイズ

### テクスチャ合成
- **texture_synthesis.py** - パッチベースのテクスチャ合成とSVG最適化の組み合わせ

### 機械学習応用
- **sketch_gan.py** - スケッチ生成のためのGANトレーニングインターフェース

## 技術的解析とテスト

### 数値解析
- **quadratic_distance_approx.py** - 二次ベジエ曲線の距離場計算における正確な計算と近似の比較
- **curve_subdivision.py** - 三次ベジエ曲線を二次ベジエ曲線に適応的分割・変換

### 評価とテスト
- **test_eval_positions.py** - 特定位置での評価機能のテスト
- **geometry.py** - SVGパスの幾何学的制約と損失関数を計算するクラス

## ビルドとユーティリティ

- **Makefile** - シームカービングデモ用のビルド設定
- **.gitignore** - Git無視ファイルの設定

## サブディレクトリ

- **generative_models/** - 生成モデル関連のコード（VAE、GAN、SketchRNN等）
- **imgs/** - デモ用のサンプル画像とSVGファイル
- **textureSyn/** - テクスチャ合成関連のコード

## 主な技術的特徴

1. **微分可能レンダリング** - 全てのデモでベクターグラフィックスの微分可能レンダリングを活用
2. **勾配ベース最適化** - Adam、SGDなどのオプティマイザーを使用した効率的な最適化
3. **多様な出力形式** - 通常のカラー描画、SDF、アルファブレンディングに対応
4. **フレームワーク対応** - PyTorchとTensorFlowの両方をサポート
5. **可視化** - Visdomを使用したリアルタイム最適化進行状況の可視化
6. **動画出力** - FFmpegを使用した最適化過程の動画生成

このライブラリは、コンピュータグラフィックス、機械学習、画像処理の研究と教育に幅広く活用できる包括的なツールセットを提供しています。
