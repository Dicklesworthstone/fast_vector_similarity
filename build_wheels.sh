#!/bin/bash
set -e -x

# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Python versions to build for
PYTHON_VERSIONS=("cp36-cp36m" "cp37-cp37m" "cp38-cp38" "cp39-cp39" "cp310-cp310" "cp311-cp311")

# Compile wheels
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    PYBIN="/opt/python/${PYVER}/bin"
    "${PYBIN}/pip" install --upgrade pip
    "${PYBIN}/pip" install maturin
    "${PYBIN}/maturin" build --release
done

# Copy the wheels to the shared volume
cp -r target/wheels /io