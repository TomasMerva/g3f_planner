#!/bin/bash

echo "Configuring the build system..."
cmake -S . -B build

echo "Building the project..."
make -C build -j4