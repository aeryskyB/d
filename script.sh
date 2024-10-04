#!/bin/bash

cd mnist

# from one of the archived pages for Lecun's MNIST dataset: https://web.archive.org/web/20070206154307/http://yann.lecun.com/exdb/mnist/
echo "Downloading train images..."
wget -nv --show-progress https://web.archive.org/web/20070206154307/http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
echo "Downloading train labels..."
wget -nv --show-progress https://web.archive.org/web/20070206154307/http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

echo "Downloading test images..."
wget -nv --show-progress https://web.archive.org/web/20070206154307/http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
echo "Downloading test labels..."
wget -nv --show-progress https://web.archive.org/web/20070206154307/http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "Done downloading ;)"
echo ""

echo "Converting to .npy objects..."
python read.py
echo "Done conversion ;)"
echo ""

echo "Deleting base files..."
rm *idx?-ubyte.gz
cd ..
echo "All done! ;)"
