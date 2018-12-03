#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Download CORnet network definitions"
mkdir -p nets/cornet
pushd nets
pushd cornet
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/__init__.py -O __init__.py
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_z.py -O cornet_z.py
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_r.py -O cornet_r.py
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_s.py -O cornet_s.py
popd
popd

echo "Download CORnet (Z, R, and S) trained on ImageNet"
mkdir -p zoo
pushd zoo
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_z_epoch25.pth.tar -O cornet_z_epoch25.pth.tar
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_r_epoch25.pth.tar -O cornet_r_epoch25.pth.tar
wget --progress=bar \
   http://web.mit.edu/sjamieso/www/6.861/cornet/cornet_s_epoch43.pth.tar -O cornet_s_epoch43.pth.tar
popd

echo "done"
