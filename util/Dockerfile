FROM centos:7

RUN yum install centos-release-scl -y
RUN yum install devtoolset-10-gcc rh-python38 rh-git227 -y
RUN source scl_source enable rh-python38 && \
    pip install twine build auditwheel setuptools patchelf autopep8 -U
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN echo "#!/bin/bash" > /etc/profile.d/enable_scl_all.sh && \
    echo "source scl_source enable devtoolset-10" >> /etc/profile.d/enable_scl_all.sh && \
    echo "source scl_source enable rh-python38" >> /etc/profile.d/enable_scl_all.sh && \
    echo "source scl_source enable rh-git227" >> /etc/profile.d/enable_scl_all.sh && \
    echo "source /root/.cargo/env" >> /etc/profile.d/enable_scl_all.sh && \
    chmod +x /etc/profile.d/enable_scl_all.sh