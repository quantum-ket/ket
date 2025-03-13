# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

FROM almalinux:8-minimal AS rust
RUN microdnf install gcc -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain 1.85 -y

FROM rust AS build_libket_amd64
COPY src/ket/clib/libs/libket/ /workdir/
WORKDIR /workdir/
RUN . "$HOME/.cargo/env" && cargo build --release

FROM rust AS build_kbw_amd64
COPY src/ket/clib/libs/kbw/ /workdir/
WORKDIR /workdir/
RUN . "$HOME/.cargo/env" && cargo build --release

FROM python:3.11-slim AS package_amd64
RUN pip install build auditwheel patchelf
WORKDIR /workdir
COPY setup.cfg setup.py README.md LICENSE MANIFEST.in ./
COPY src/ket/ src/ket/
RUN rm -rf src/ket/clib/libs/libket src/ket/clib/libs/kbw
COPY --from=build_libket_amd64 /workdir/target/release/libket.so src/ket/clib/libs/libket.so
COPY --from=build_kbw_amd64 /workdir/target/release/libkbw.so src/ket/clib/libs/libkbw.so
RUN python -m build -w
RUN python -m auditwheel repair --plat manylinux_2_28_x86_64 dist/ket_lang*.whl

FROM busybox
COPY --from=package_amd64 /workdir/wheelhouse/* .
