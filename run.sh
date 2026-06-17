#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <filename_without_.cu>"
    echo "Example: ./run.sh 07_half2_vectorized"
    exit 1
fi

FILE=$1

mkdir -p build

echo "Compiling src/${FILE}.cu ..."
nvcc -O2 -std=c++17 -lineinfo \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    src/${FILE}.cu -o build/${FILE}

echo "Running build/${FILE} ..."
./build/${FILE}