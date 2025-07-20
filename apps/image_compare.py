"""
画像比較ツール

このスクリプトは2つの画像を参照画像と比較し、
各種画像品質メトリクス（MSE、PSNR、SSIM）を計算します。
また、差分画像を可視化して保存します。

使用例:
    python image_compare.py img1.png img2.png reference.png
"""

import argparse
import skimage.io
import numpy as np
from matplotlib import cm
import math
from skimage.metrics import structural_similarity as ssim

def normalize(x, min_, max_):
    """
    値を[0, 1]の範囲に正規化します。
    
    引数:
        x: 正規化する配列
        min_: 最小値
        max_: 最大値
        
    戻り値:
        正規化された配列
    """
    return (x - min_) / (max_ - min_)

def main(args):
    """
    画像比較のメイン処理を実行します。
    
    引数:
        args: コマンドライン引数
    """
    # 画像を読み込み、float32形式に変換
    img1 = skimage.img_as_float(skimage.io.imread(args.img1)).astype(np.float32)
    img2 = skimage.img_as_float(skimage.io.imread(args.img2)).astype(np.float32)
    ref = skimage.img_as_float(skimage.io.imread(args.ref)).astype(np.float32)
    
    # RGBチャンネルのみを使用（アルファチャンネルを除去）
    img1 = img1[:, :, :3]
    img2 = img2[:, :, :3]
    ref = ref[:, :, :3]

    # 参照画像との絶対差分を計算
    diff1 = np.sum(np.abs(img1 - ref), axis = 2)
    diff2 = np.sum(np.abs(img2 - ref), axis = 2)
    
    # 差分の正規化範囲を決定
    min_ = min(np.min(diff1), np.min(diff2))
    max_ = max(np.max(diff1), np.max(diff2)) * 0.5  # 可視化のため最大値を調整
    
    # カラーマップを適用して差分を可視化
    diff1 = cm.viridis(normalize(diff1, min_, max_))
    diff2 = cm.viridis(normalize(diff2, min_, max_))

    # === 画像品質メトリクスの計算 ===
    
    # MSE（平均二乗誤差）
    print('MSE img1:', np.mean(np.power(img1 - ref, 2.0)))
    print('MSE img2:', np.mean(np.power(img2 - ref, 2.0)))
    
    # PSNR（ピーク信号対雑音比）
    print('PSNR img1:', 20 * math.log10(1.0 / math.sqrt(np.mean(np.power(img1 - ref, 2.0)))))
    print('PSNR img2:', 20 * math.log10(1.0 / math.sqrt(np.mean(np.power(img2 - ref, 2.0)))))
    
    # SSIM（構造的類似性指標）
    print('SSIM img1:', ssim(img1, ref, multichannel=True))
    print('SSIM img2:', ssim(img2, ref, multichannel=True))

    # 差分画像を保存
    skimage.io.imsave('diff1.png', (diff1 * 255).astype(np.uint8))
    skimage.io.imsave('diff2.png', (diff2 * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2つの画像を参照画像と比較し、品質メトリクスを計算")
    parser.add_argument("img1", help="比較対象画像1のパス")
    parser.add_argument("img2", help="比較対象画像2のパス")
    parser.add_argument("ref", help="参照画像のパス")
    args = parser.parse_args()
    main(args)
