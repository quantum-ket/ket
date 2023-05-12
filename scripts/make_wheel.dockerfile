FROM rust:1.69-slim-buster AS build_libket_amd64
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
COPY src/ket/clib/libs/libket/ .
RUN cargo build --release

FROM rust:1.69-slim-buster AS build_libket_aarch64
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
RUN apt update && apt install -y gcc-aarch64-linux-gnu
RUN rustup target add aarch64-unknown-linux-gnu
COPY src/ket/clib/libs/libket/ .
RUN CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc cargo build --target aarch64-unknown-linux-gnu --release

FROM rust:1.69-slim-buster AS build_kbw_amd64
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
COPY src/ket/clib/libs/kbw/ .
RUN cargo build --release

FROM rust:1.69-slim-buster AS build_kbw_aarch64
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
RUN apt update && apt install -y gcc-aarch64-linux-gnu
RUN rustup target add aarch64-unknown-linux-gnu
COPY src/ket/clib/libs/kbw/ .
RUN CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc cargo build --target aarch64-unknown-linux-gnu --release

FROM python:3-slim AS package_amd64
RUN pip install build auditwheel patchelf
WORKDIR /workdir
COPY setup.cfg setup.py README.md LICENSE MANIFEST.in ./
COPY src/ket/ src/ket/
RUN rm -rf src/ket/clib/libs/libket src/ket/clib/libs/kbw
COPY --from=build_libket_amd64 target/release/libket.so src/ket/clib/libs/libket.so
COPY --from=build_kbw_amd64 target/release/libkbw.so src/ket/clib/libs/libkbw.so
RUN python -m build -w
RUN python -m auditwheel repair --plat manylinux_2_28_x86_64 dist/ket_lang*.whl

FROM --platform=arm64 python:3-slim AS package_aarch64
RUN pip install build auditwheel patchelf
WORKDIR /workdir
COPY setup.cfg setup.py README.md LICENSE MANIFEST.in ./
COPY src/ket/ src/ket/
RUN rm -rf src/ket/clib/libs/libket src/ket/clib/libs/kbw
COPY --from=build_libket_aarch64 target/aarch64-unknown-linux-gnu/release/libket.so src/ket/clib/libs/libket.so
COPY --from=build_kbw_aarch64 target/aarch64-unknown-linux-gnu/release/libkbw.so src/ket/clib/libs/libkbw.so
RUN python -m build -w
RUN python -m auditwheel repair --plat manylinux_2_28_aarch64 dist/ket_lang*.whl

FROM busybox
COPY --from=package_amd64 /workdir/wheelhouse/* .
COPY --from=package_aarch64 /workdir/wheelhouse/* .
