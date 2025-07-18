## 環境構築手順
- 環境: Ubuntu 24.04
- Python: 3.9
- CUDA: 12.0
- 仮想環境: `uv` 
※ 注意
- colabでできるかは不明、ローカルはdiffvgのビルドがめんどくさそう、windowsやMacの挙動が不明
- diffvgはCPUだけ使うけど、ImageRewardはGPUがないと動かない可能性がある

```bash
git clone https://github.com/bishopfunc/diffvg.git
cd diffvg
uv sync # これで仮想環境が作成される
source .venv/bin/activate # 仮想環境をアクティブ化
python setup.py install # diffvgパッケージのインストール
python font_sample/main.py # サンプルを実行
```
※ 公式との変更点
- `python setup.py install`でCUDAのコンパイルが失敗したため、CPUでコンパイルしている
  - `seutup.py`の`build_with_cuda = False`を変更
- そのため、使用時は`pydiffvg.set_use_gpu(False)`に設定する必要がある

## 作業中のファイル
- `memo.md`: このメモファイル
- `assets/images`: ImageRewardで使用するサンプル画像を保存するディレクトリ
- `font_sample`: 作業用のディレクトリ
- `font_sample/main.py`: [この記事のサンプル](https://zenn.dev/morisawa/articles/about-diffvg)
- `font_sample/main_font.py`: [この記事のサンプル(フォント) + ImageReward](https://zenn.dev/morisawa/articles/about-diffvg_font)
- `font_sample/test_image_reward.py`: [ImageReward](https://github.com/THUDM/ImageReward/tree/main?tab=readme-ov-file)のサンプル
- `loss = diffvg_loss - reward`でImageRewardを追加して最適化するのを意図している
- `font_sample/target_image.png`: 最適化対象の画像に`results/font_sample/target_image.png`をコピーして使用
- `font_sample/temp_output.svg`: フォントのSVGを確認するために出力したもの、実際は文字列でフォントの種類を指定している
- `results`: 最適化結果を保存するディレクトリ、`results/font_sample`にフォントサンプルの結果が保存される


## TODO
- ゴール: ベクトル画像の最適化にテキストの情報を加える、そのアイデアの検証
- 使用技術
  - diffvg: フォントの最適化
  - ImageReward: 言語モデルを使った画像の評価
- 良さげ動画を複数個取れる + 最適化っぽい問題設定を考えればOKではないか

問題
- diffvgnの最適化がうまくいってない
- 多分選んでいるフォントが良くない、目標
- フォントじゃなくて、図形の最適化でもいいのかも、サンプルはたくさんある。`apps/*.py`を参考にする。
- `font_sample/out.mp4`を見ると、発散?してることがわかる


TODO:
- [ ] フォントと目標画像の選定を見直す
- [ ] diffvgライブラリの使い方が正しいか確認
- [ ] diffvgの最適化単体でうまくいった場合、ImageRewardのプロンプトを見直す
- [ ] ImageRewardのrewardのスケーリング
- [ ] diffvgの仕組みの理解
- [ ] ImageRewardはブラックボックスなので、それにあった最適化アルゴリズムを選定しても良いかも(発展的な話)


## トラブルシューティングメモ
個人的なメモなので気にしないでください。
```bash
uv add torch torchvision numpy scikit-image cssutils svgpathtools
uv add "setuptools==68.0"
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
rm -rf build
uv add pybind11
python setup.py install
```



```bash
error: expected template-name before ‘<’ token
error: expected identifier before ‘<’ token
error: expected primary-expression before ‘)’ token
```


```bash
#error -- unsupported GNU version! gcc versions later than 12 are not supported!
```

pybind11のバージョンを固定する必要がある
```bash
cd pybind11
git checkout v2.13.6
git submodule update --init --recursive
git describe --tags

git rm --cached pybind11
rm -rf pybind11
rm -rf .git/modules/pybind11

git submodule add https://github.com/pybind/pybind11.git pybind11
git submodule update --init --recursive
```

camkeの設定
```bash
rm -rf build
mkdir build && cd build
cmake .. -DPython_EXECUTABLE=$(which python3)
```



potryでもダメだった
CPUでコンパイル
build_with_cuda = False

