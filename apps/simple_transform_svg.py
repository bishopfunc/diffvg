"""
SVG変換最適化デモ

このスクリプトはSVGファイルを読み込み、目標画像に向けて
アフィン変換パラメータを最適化します。
Visdomを使用してリアルタイムで最適化の進行状況を可視化し、
ガウシアンスムージングと勾配計算を組み合わせた高度な最適化を行います。
"""

import pydiffvg
import torch
import torchvision
from PIL import Image
import numpy as np

# GPUが利用可能な場合は使用
pydiffvg.set_use_gpu(torch.cuda.is_available())

def inv_exp(a, x, xpow=1):
    """
    逆指数関数を計算します。
    
    引数:
        a: 底
        x: 指数の基数
        xpow: 指数の累乗
        
    戻り値:
        計算結果
    """
    return pow(a, pow(1.-x, xpow))

import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

import visdom

class GaussianSmoothing(nn.Module):
    """
    1次元、2次元、3次元テンソルにガウシアンスムージングを適用します。
    
    入力の各チャンネルに対して深度方向の畳み込みを使用して
    フィルタリングを個別に実行します。
    
    引数:
        channels (int, sequence): 入力テンソルのチャンネル数。
                                 出力も同じチャンネル数になります。
        kernel_size (int, sequence): ガウシアンカーネルのサイズ。
        sigma (float, sequence): ガウシアンカーネルの標準偏差。
        dim (int, optional): データの次元数。デフォルトは2（空間）。
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # ガウシアンカーネルは各次元のガウシアン関数の積
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # ガウシアンカーネルの値の合計が1になるように正規化
        kernel = kernel / torch.sum(kernel)

        # 深度方向畳み込み重みの形状に変形
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        # 次元に応じて適切な畳み込み関数を選択
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'サポートされているのは1、2、3次元のみです。受信: {}'.format(dim)
            )

    def forward(self, input):
        """
        入力にガウシアンフィルターを適用します。
        
        引数:
            input (torch.Tensor): ガウシアンフィルターを適用する入力。
            
        戻り値:
            filtered (torch.Tensor): フィルター処理された出力。
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

# Visdomクライアントを初期化（ポート8080で可視化）
vis = visdom.Visdom(port=8080)

# 4チャンネル、カーネルサイズ5、標準偏差1のガウシアンスムージングを作成
smoothing = GaussianSmoothing(4, 5, 1)

# SVG最適化設定を構成
settings = pydiffvg.SvgOptimizationSettings()
settings.global_override(["optimize_color"], False)          # 色の最適化を無効
settings.global_override(["optimize_alpha"], False)          # アルファの最適化を無効
settings.global_override(["gradients", "optimize_color"], False)     # 勾配色の最適化を無効
settings.global_override(["gradients", "optimize_alpha"], False)     # 勾配アルファの最適化を無効
settings.global_override(["gradients", "optimize_stops"], False)     # 勾配ストップの最適化を無効
settings.global_override(["gradients", "optimize_location"], False)  # 勾配位置の最適化を無効
settings.global_override(["optimizer"], "Adam")              # Adamオプティマイザーを使用
settings.global_override(["paths", "optimize_points"], False)        # パス点の最適化を無効
settings.global_override(["transforms", "transform_lr"], 1e-2)       # 変換学習率を設定
settings.undefault("linearGradient3152")
settings.retrieve("linearGradient3152")[0]["transforms"]["optimize_transforms"] = False

# 最適化可能なSVGオブジェクトを作成
# optim = pydiffvg.OptimizableSvg("note_small.svg", settings, verbose=True)
optim = pydiffvg.OptimizableSvg("heart_green.svg", settings, verbose=True)

# 目標画像を読み込み
# img = torchvision.transforms.ToTensor()(Image.open("note_transformed.png")).permute(1,2,0)
img = torchvision.transforms.ToTensor()(Image.open("heart_green_90.png")).permute(1,2,0)

name = "heart_green_90"

# 目標画像を保存
pydiffvg.imwrite(img.cpu(), 'results/simple_transform_svg/target.png')
target = img.clone().detach().requires_grad_(False)

# 初期画像を描画・保存
img = optim.render()
pydiffvg.imwrite(img.cpu(), 'results/simple_transform_svg/init.png')

def smooth(input, kernel):
    """
    入力画像にガウシアンスムージングを適用します。
    
    引数:
        input: 入力画像テンソル
        kernel: ガウシアンカーネル
        
    戻り値:
        スムージング処理された画像
    """
    input = torch.nn.functional.pad(input.permute(2,0,1).unsqueeze(0), (2, 2, 2, 2), mode='reflect')
    output = kernel(input)
    return output

def printimg(optim):
    """
    最適化オブジェクトから画像を描画し、白背景と合成します。
    
    引数:
        optim: 最適化可能なSVGオブジェクト
        
    戻り値:
        白背景と合成された画像
    """
    img = optim.render()
    comp = img.clone().detach()
    bg = torch.tensor([[[1., 1., 1.]]])  # 白背景
    comprgb = comp[:, :, 0:3]            # RGBチャンネル
    compalpha = comp[:, :, 3].unsqueeze(2)  # アルファチャンネル
    # アルファブレンディング
    comp = comprgb * compalpha + bg * (1 - compalpha)
    return comp

def comp_loss_and_grad(img, tgt, it, sz):
    """
    損失と勾配を計算し、差分画像を保存します。
    
    引数:
        img: 現在の画像
        tgt: 目標画像
        it: 反復回数
        sz: 出力サイズ
        
    戻り値:
        loss: L2損失
        res: 計算された勾配
    """
    dif = img - tgt

    # L2損失を計算
    loss = dif.pow(2).mean()

    dif = dif.detach()

    # 差分の絶対値を計算し、アルファチャンネルを1に設定
    cdif = dif.clone().abs()
    cdif[:, :, 3] = 1.

    # 差分画像をリサイズして保存
    resdif = torch.nn.functional.interpolate(cdif.permute(2,0,1).unsqueeze(0), sz, mode='bilinear').squeeze().permute(1,2,0).abs()
    pydiffvg.imwrite(resdif[:, :, 0:4], 'results/simple_transform_svg/dif_{:04}.png'.format(it))

    # NumPy配列に変換して勾配計算
    dif = dif.numpy()
    padded = np.pad(dif, [(1,1), (1,1), (0,0)], mode='edge')
    
    # x方向とy方向の勾配を計算
    grad_x = (padded[:-2, :, :] - padded[2:, :, :])[:, 1:-1, :]
    grad_y = (padded[:, :-2, :] - padded[:, 2:, :])[1:-1, :, :]

    resshape = dif.shape
    resshape = (resshape[0], resshape[1], 2)
    res = np.zeros(resshape)

    # 各ピクセルで最小二乗法を使用して勾配を計算
    for x in range(resshape[0]):
        for y in range(resshape[1]):
            A = np.concatenate((grad_x[x, y, :][:, np.newaxis], grad_y[x, y, :][:, np.newaxis]), axis=1)
            b = -dif[x, y, :]
            v = np.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, b))
            res[x, y, :] = v[0]

    return loss, res

import colorsys

def print_gradimg(gradimg, it, shape=None):
    """
    勾配画像を可視化して保存します。
    
    引数:
        gradimg: 勾配画像
        it: 反復回数
        shape: 出力形状（オプション）
    """
    out = torch.zeros((gradimg.shape[0], gradimg.shape[1], 3), requires_grad=False, dtype=torch.float32)
    for x in range(gradimg.shape[0]):
        for y in range(gradimg.shape[1]):
            h = math.atan2(gradimg[x, y, 1], gradimg[x, y, 0])
            s = math.tanh(np.linalg.norm(gradimg[x, y, :]))
            v = 1.
            vec = (gradimg[x, y, :].clip(min=-1, max=1) / 2) + 0.5
            # out[x, y, :] = torch.tensor(colorsys.hsv_to_rgb(h, s, v), dtype=torch.float32)
            out[x, y, :] = torch.tensor([vec[0], vec[1], 0])

    # 指定された形状にリサイズ
    if shape is not None:
        out = torch.nn.functional.interpolate(out.permute(2,0,1).unsqueeze(0), shape, mode='bilinear').squeeze().permute(1,2,0)
    pydiffvg.imwrite(out.cpu(), 'results/simple_transform_svg/grad_{:04}.png'.format(it))

# 1000回のAdam反復を実行
for t in range(1000):
    print('iteration:', t)
    optim.zero_grad()
    
    # 中間SVGファイルを保存
    with open('results/simple_transform_svg/viter_{:04}.svg'.format(t), "w") as f:
        f.write(optim.write_xml())
    
    # スケール計算（現在は未使用）
    scale = inv_exp(1/16, math.pow(t/1000, 1), 0.5)
    # print(scale)
    # img = optim.render(seed=t+1, scale=scale)
    img = optim.render(seed=t + 1, scale=None)
    
    # Visdomで画像サイズをプロット
    vis.line(torch.tensor([img.shape[0]]), X=torch.tensor([t]), win=name + " size", update="append",
             opts={"title": name + " size"})
    # print(img.shape)
    # img = optim.render(seed=t + 1)

    # 目標画像を現在の画像サイズにリサイズ
    ptgt = target.permute(2,0,1).unsqueeze(0)
    sz = img.shape[0:2]
    restgt = torch.nn.functional.interpolate(ptgt, size=sz, mode='bilinear').squeeze().permute(1,2,0)

    # 損失関数を計算（L2損失）
    # loss = (smooth(img, smoothing) - smooth(restgt, smoothing)).pow(2).mean()
    # loss = (img - restgt).pow(2).mean()
    # loss = (img - target).pow(2).mean()
    loss, gradimg = comp_loss_and_grad(img, restgt, t, target.shape[0:2])
    print_gradimg(gradimg, t, target.shape[0:2])
    print('loss:', loss.item())
    
    # Visdomで損失をプロット
    vis.line(loss.unsqueeze(0), X=torch.tensor([t]), win=name + " loss", update="append",
             opts={"title": name + " loss"})

    # 勾配を逆伝播
    loss.backward()

    # 勾配降下ステップを実行
    optim.step()

    # 中間描画結果を保存
    comp = printimg(optim)
    pydiffvg.imwrite(comp.cpu(), 'results/simple_transform_svg/iter_{:04}.png'.format(t))


# 最終結果を描画

img = optim.render()
# 最終画像とSVGファイルを保存
pydiffvg.imwrite(img.cpu(), 'results/simple_transform_svg/final.png')
with open('results/simple_transform_svg/final.svg', "w") as f:
    f.write(optim.write_xml())

# 中間描画結果を動画に変換
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/simple_transform_svg/iter_%04d.png", "-vb", "20M",
    "results/simple_transform_svg/out.mp4"])

# 勾配画像も動画に変換
call(["ffmpeg", "-framerate", "24", "-i",
    "results/simple_transform_svg/grad_%04d.png", "-vb", "20M",
    "results/simple_transform_svg/out_grad.mp4"])
