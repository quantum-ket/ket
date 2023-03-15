#/bin/bash
docker build -t ket_wheel -f scripts/make.dockerfile .
docker run -v $PWD:/mnt --rm ket_wheel /bin/bash -c "mkdir -p /mnt/wheelhouse && cp ket*.whl /mnt/wheelhouse"
