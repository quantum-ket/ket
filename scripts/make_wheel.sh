#!/bin/sh
docker build -t ket_wheel -f scripts/make_wheel.dockerfile .
docker run -v $PWD:/mnt --rm ket_wheel /bin/sh -c "mkdir -p /mnt/wheelhouse && cp ket*.whl /mnt/wheelhouse"
