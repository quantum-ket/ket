#!/bin/sh

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

docker build -t ket_wheel:linux -f scripts/wheel-linux-x86_64.dockerfile .
docker run -v $PWD:/mnt --rm ket_wheel:linux /bin/sh -c "mkdir -p /mnt/wheelhouse && cp ket*.whl /mnt/wheelhouse"
