```bash
uv add torch torchvision numpy scikit-image cssutils svgpathtools
uv add "setuptools==68.0"
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
rm -rf build
uv add pybind11
python setup.py install
```

error: expected template-name before ‘<’ token
error: expected identifier before ‘<’ token
error: expected primary-expression before ‘)’ token


#error -- unsupported GNU version! gcc versions later than 12 are not supported!



git rm --cached pybind11
rm -rf pybind11
rm -rf .git/modules/pybind11

git submodule add https://github.com/pybind/pybind11.git pybind11
git submodule update --init --recursive

cd pybind11
git checkout v2.13.6
git submodule update --init --recursive
git describe --tags

rm -rf build
mkdir build && cd build
cmake .. -DPython_EXECUTABLE=$(which python3)


set LIBDIR=./.venv/lib/python3.9/site-packages


potryでもダメだった
CPUでコンパイル
build_with_cuda = False


ln -s /mnt/a/yamamoto/diffvg/results /home/yamamoto/workspace/diffvg/results