#!/bin/sh

python -m build
auditwheel repair --plat manylinux_2_17_x86_64 dist/ket_lang-`cat ket/__version__.py | awk -F\' '{print $2}'`-py3-none-any.whl
