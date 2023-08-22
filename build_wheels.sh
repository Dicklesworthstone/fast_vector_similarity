#!/bin/bash
set -e -x

# Compile wheels
for PYBIN in /opt/python/cp3[6-9]*; do
    "${PYBIN}/bin/pip" install --upgrade pip
    "${PYBIN}/bin/pip" install maturin
    "${PYBIN}/bin/maturin" build --release
done

# Copy the wheels to the shared volume
cp -r target/wheels /io
