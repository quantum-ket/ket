# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

FROM registry.gitlab.com/quantum-ket/ket-building-images:windows-1.88 AS build_libket_amd64
COPY src/ket/clib/libs/libket/ /workdir/
WORKDIR /workdir/
RUN cargo build --release --target x86_64-pc-windows-gnu

FROM registry.gitlab.com/quantum-ket/ket-building-images:windows-1.88 AS build_kbw_amd64
COPY src/ket/clib/libs/kbw/ /workdir/
WORKDIR /workdir/
RUN cargo build --release --target x86_64-pc-windows-gnu

FROM ghcr.io/astral-sh/uv:0.13.5-python3.11-trixie-slim AS package_amd64
WORKDIR /workdir
COPY pyproject.toml README.md LICENSE ./
COPY src/ket/ src/ket/
RUN rm -rf src/ket/clib/libs/libket src/ket/clib/libs/kbw
COPY --from=build_libket_amd64 /workdir/target/x86_64-pc-windows-gnu/release/ket.dll src/ket/clib/libs/ket.dll
COPY --from=build_kbw_amd64 /workdir/target/x86_64-pc-windows-gnu/release/kbw.dll src/ket/clib/libs/kbw.dll
RUN uv build --wheel
RUN uvx wheel tags --platform-tag win_amd64 --remove dist/ket_lang*.whl

FROM busybox
COPY --from=package_amd64 /workdir/dist/* .
