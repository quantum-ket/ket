#!/bin/sh

mkdir -p libs 
cd libs 
git clone https://gitlab.com/quantum-ket/libket.git
git clone https://gitlab.com/quantum-ket/kbw.git

cd libket
cargo build --release
cd ../kbw
cargo build --release
cd ../..

mkdir -p ket/clib/libs
mv libs/libket/target/release/libket.so ket/clib/libs/
mv libs/kbw/target/release/libkbw.so ket/clib/libs/