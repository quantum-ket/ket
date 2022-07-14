FROM centos:7

RUN yum install centos-release-scl -y
RUN yum install devtoolset-11-gcc rh-python38 rh-git227 -y
RUN source scl_source enable rh-python38 && \
    pip install twine build auditwheel setuptools patchelf -U
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="${HOME}/.cargo/env:${PATH}"
