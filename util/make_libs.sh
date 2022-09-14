#!/bin/sh

mkdir -p libs 
cd libs 
git clone https://gitlab.com/quantum-ket/libket.git || git -C libket pull
git clone https://gitlab.com/quantum-ket/kbw.git || git -C kbw pull

cd libket
cargo build --release
cd ../kbw
cargo build --release
cd ../..
