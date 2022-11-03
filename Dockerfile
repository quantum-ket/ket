FROM centos:7

ARG USERNAME=code
ARG USER_UID=1000
ARG USER_GID=$USER_UID

COPY requirements_dev.txt .

RUN yum install centos-release-scl -y
RUN yum install devtoolset-10-gcc rh-python38 rh-git227 sudo zsh -y
RUN source scl_source enable rh-python38 && \
    pip install auditwheel patchelf setuptools -U && \
    pip install -r requirements_dev.txt
RUN echo "#!/bin/bash" > /etc/profile.d/enable_scl_all.sh && \
    echo "source scl_source enable devtoolset-10" >> /etc/profile.d/enable_scl_all.sh && \
    echo "source scl_source enable rh-python38" >> /etc/profile.d/enable_scl_all.sh && \
    echo "source scl_source enable rh-git227" >> /etc/profile.d/enable_scl_all.sh && \
    echo "source /home/code/.cargo/env" >> /etc/profile.d/enable_scl_all.sh && \
    chmod +x /etc/profile.d/enable_scl_all.sh
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd -s /usr/bin/zsh --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN source scl_source enable rh-git227 && \
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" -s --batch
