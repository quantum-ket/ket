#!/bin/sh

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

docker build -t ket_wheel:linux -f scripts/make_wheel_linux-x86_64.dockerfile .
docker build -t ket_wheel:windows -f scripts/make_wheel_windows-x86_64.dockerfile .
docker run -v $PWD:/mnt --rm ket_wheel:linux /bin/sh -c "mkdir -p /mnt/wheelhouse && cp ket*.whl /mnt/wheelhouse"
docker run -v $PWD:/mnt --rm ket_wheel:windows /bin/sh -c "mkdir -p /mnt/wheelhouse && cp ket*.whl /mnt/wheelhouse"
