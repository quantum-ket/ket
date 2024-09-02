# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

FROM rust:1.79-slim-buster AS build_libket_aarch64
RUN apt update && apt install -y gcc-aarch64-linux-gnu
RUN rustup target add aarch64-unknown-linux-gnu
COPY src/ket/clib/libs/libket/ /workdir/
WORKDIR /workdir/
RUN CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc cargo build --target aarch64-unknown-linux-gnu --release

FROM rust:1.79-slim-buster AS build_kbw_aarch64
RUN apt update && apt install -y gcc-aarch64-linux-gnu
RUN rustup target add aarch64-unknown-linux-gnu
COPY src/ket/clib/libs/kbw/ /workdir/
WORKDIR /workdir/
RUN CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc cargo build --target aarch64-unknown-linux-gnu --release

FROM --platform=arm64 python:3-slim AS package_aarch64
RUN pip install build auditwheel patchelf
WORKDIR /workdir
COPY setup.cfg setup.py README.md LICENSE MANIFEST.in ./
COPY src/ket/ src/ket/
RUN rm -rf src/ket/clib/libs/libket src/ket/clib/libs/kbw
COPY --from=build_libket_aarch64 /workdir/target/aarch64-unknown-linux-gnu/release/libket.so src/ket/clib/libs/libket.so
COPY --from=build_kbw_aarch64 /workdir/target/aarch64-unknown-linux-gnu/release/libkbw.so src/ket/clib/libs/libkbw.so
RUN python -m build -w
RUN python -m auditwheel repair --plat manylinux_2_28_aarch64 dist/ket_lang*.whl

FROM busybox
COPY --from=package_aarch64 /workdir/wheelhouse/* .
