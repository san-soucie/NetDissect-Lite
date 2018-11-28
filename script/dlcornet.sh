#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Download CORnet (Z, R, and S) trained on ImageNet"
#echo "Downloading $MODEL"
mkdir -p zoo
pushd zoo
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_z_epoch25.pth.tar
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_r_epoch25.pth.tar
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_s_epoch43.pth.tar
popd

mkdir -p nets/cornet
pushd nets
pushd cornet
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/__init__.py
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_z.py
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_r.py
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_s.py
popd
popd

echo "done"
