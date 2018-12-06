#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Download broden1_224
if [[ ! -f dataset/tiny-imagenet-200/test/images/test_0.JPEG]]
then

echo "Downloading tiny-imagenet-200"
mkdir -p dataset
pushd dataset
wget --progress=bar \
   http://cs231n.stanford.edu/tiny-imagenet-200.zip \
   -O tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
popd

fi