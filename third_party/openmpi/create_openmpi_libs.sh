#!/bin/bash

# Configure OpenMPI build
mkdir -p build/install
cd build
if [ $? -ne 0 ]; then
    echo "Failed to create or change to build directory."
    exit 1
fi

../configure --prefix=build/install CC=clang-18 CXX=clang++-18 FC=flang-7
# Check if the configuration was successful
if [ $? -ne 0 ]; then
    echo "Configuration failed. Please check the output for errors."
    exit 1
fi
