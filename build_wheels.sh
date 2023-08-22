#!/bin/bash
set -e -x

# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install Python dependencies
yum install -y gcc make libffi-devel openssl-devel bzip2-devel zlib-devel readline-devel sqlite-devel xz-devel ncurses-devel gdbm-devel

# Install pyenv
curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Python versions to build for
PYTHON_VERSIONS=("3.9.7" "3.10.0" "3.11.0")

# Compile wheels
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    pyenv install $PYVER
    pyenv global $PYVER
    python -m pip install --upgrade pip
    pip install maturin
    maturin build --release
done

# Copy the wheels to the shared volume
cp -r target/wheels /io
