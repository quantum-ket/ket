#!/bin/sh

python -m build
auditwheel repair --plat manylinux_2_17_x86_64 dist/ket_lang-*-py3-none-any.whl
