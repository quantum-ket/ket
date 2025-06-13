# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

FROM rust:1.87 AS rust
RUN apt update && apt install mingw-w64 -y
RUN rustup target add x86_64-pc-windows-gnu

FROM rust AS build_libket_amd64
COPY src/ket/clib/libs/libket/ /workdir/
WORKDIR /workdir/
RUN cargo build --release --target x86_64-pc-windows-gnu

FROM rust AS build_kbw_amd64
COPY src/ket/clib/libs/kbw/ /workdir/
WORKDIR /workdir/
RUN cargo build --release --target x86_64-pc-windows-gnu

FROM python:3.11-slim AS package_amd64
RUN pip install build auditwheel patchelf setuptools
WORKDIR /workdir
COPY setup.cfg setup.py README.md LICENSE MANIFEST.in ./
COPY src/ket/ src/ket/
RUN rm -rf src/ket/clib/libs/libket src/ket/clib/libs/kbw
COPY --from=build_libket_amd64 /workdir/target/x86_64-pc-windows-gnu/release/ket.dll src/ket/clib/libs/ket.dll
COPY --from=build_kbw_amd64 /workdir/target/x86_64-pc-windows-gnu/release/kbw.dll src/ket/clib/libs/kbw.dll
RUN python setup.py bdist_wheel -p win-amd64

FROM busybox
COPY --from=package_amd64 /workdir/dist/* .
