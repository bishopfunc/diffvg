# diffvgアプリケーション概要

このドキュメントは`apps`ディレクトリ内の全てのPythonファイルの役割と機能をまとめたものです。

## 基本デモ・チュートリアル

### 単一形状の最適化デモ
- **single_circle.py**: 単一円の最適化デモ。目標円に向けて半径、中心位置、色を勾配降下法で最適化
- **single_circle_outline.py**: ストローク付き円の最適化。塗りつぶしとストロークの両方のパラメータを最適化
- **single_circle_sdf.py**: SDF（符号付き距離場）出力を使った円の最適化デモ
- **single_circle_tf.py**: TensorFlow版の円最適化デモ
- **single_rect.py**: 単一矩形の最適化。位置、サイズ、色を最適化
- **single_ellipse.py**: 楕円の最適化。半径、中心位置、色を最適化
- **single_ellipse_transform.py**: アフィン変換を使った楕円の最適化デモ
- **single_polygon.py**: 多角形の最適化。頂点位置と色を最適化

### ベジエ曲線・パスの最適化
- **single_curve.py**: 複数セグメントのベジエ曲線最適化。閉じたパスの制御点と色を最適化
- **single_curve_outline.py**: ストローク付きベジエ曲線の最適化
- **single_curve_sdf.py**: SDF出力を使ったベジエ曲線の最適化
- **single_curve_sdf_trans.py**: SDF変換とグラデーション計算のテスト用スクリプト
- **single_curve_tf.py**: TensorFlow版のベジエ曲線最適化
- **single_open_curve.py**: 開いた曲線（ストロークのみ）の最適化
- **single_open_curve_thickness.py**: 可変線幅を持つ開いた曲線の最適化
- **single_path.py**: 複雑なSVGパス（飛行機形状）の最適化
- **single_path_sdf.py**: SDF出力を使った複雑パスの最適化
- **single_stroke.py**: ストロークのみのパス最適化
- **single_stroke_tf.py**: TensorFlow版のストローク最適化

### グラデーション・フィルター
- **single_gradient.py**: 線形グラデーションの最適化。グラデーションの開始点、終了点、色を最適化
- **optimize_pixel_filter.py**: ピクセルフィルター（Hannフィルター）の半径パラメータ最適化デモ

## 高度なアプリケーション

### 画像近似・スタイル変換
- **painterly_rendering.py**: 絵画風レンダリング。入力画像をランダム生成パスで近似し、絵画的表現を生成
- **refine_svg.py**: 既存SVGファイルの最適化。目標画像に向けて制御点と色を調整
- **style_transfer.py**: ニューラルスタイル転送。VGG19を使った知覚損失でSVGのスタイル変換
- **texture_synthesis.py**: テクスチャ合成。パッチベースのテクスチャ合成とSVG最適化の組み合わせ

### 画像処理・変換
- **gaussian_blur.py**: ガウシアンブラー効果の最適化。ぼかした目標画像に向けてベクターパラメータを調整
- **seam_carving.py**: シームカービングによるSVGリターゲティング。エネルギー関数に基づく画像縮小
- **render_svg.py**: SVGからPNGへの変換ユーティリティ

### 機械学習・生成モデル
- **sketch_gan.py**: スケッチ生成GAN。MNISTデータセットを使ったベクタースケッチ生成
- **simple_transform_svg.py**: SVG変換の最適化。Visdomを使った可視化付き

## 解析・比較ツール

### 勾配・数値解析
- **finite_difference_comp.py**: 有限差分法と解析的勾配の比較。数値勾配と解析勾配の精度検証
- **shared_edge_compare.py**: 共有エッジを持つ形状の勾配比較。前進差分、後退差分、解析的勾配の比較
- **quadratic_distance_approx.py**: 二次距離近似の比較。正確な距離計算と近似手法の精度評価

### 画像品質評価
- **image_compare.py**: 画像比較ツール。MSE、PSNR、SSIM等の画像品質メトリクス計算

### 幾何学・曲線解析
- **curve_subdivision.py**: 三次ベジエ曲線の二次ベジエ曲線への適応的分割・変換
- **geometry.py**: SVGパスの幾何学的制約（水平/垂直配置、平行線、滑らかさ）を扱う損失関数クラス

## インタラクティブツール

- **svg_brush.py**: インタラクティブなSVGブラシツール。マウス操作でSVGを編集
- **test_eval_positions.py**: 評価位置指定機能のテスト。特定位置でのSDF値評価

## パーサー・テストツール

- **svg_parse_test.py**: SVGパーサーのテスト。SVG読み込みと再保存の検証

## ビルド・設定ファイル

- **Makefile**: シームカービングの自動化ビルドファイル
- **.gitignore**: Git除外設定

## 主要な技術要素

### レンダリング技術
- **微分可能レンダリング**: 全てのアプリケーションで使用される核心技術
- **SDF（符号付き距離場）**: 形状の距離情報を活用した最適化
- **アンチエイリアシング**: 高品質な画像生成のためのサンプリング

### 最適化手法
- **勾配降下法**: Adam、SGDオプティマイザーを使った最適化
- **知覚損失**: LPIPS等の知覚的画像品質評価
- **幾何学的制約**: 形状の幾何学的性質を保持する制約

### 出力形式
- **PNG画像**: 最適化過程と最終結果の可視化
- **SVGファイル**: ベクター形式での結果保存
- **MP4動画**: 最適化過程のアニメーション

このライブラリは微分可能ベクターグラフィックスの包括的なデモンストレーションを提供し、
研究から実用的なアプリケーションまで幅広い用途に対応しています。
