#!/bin/bash
# Concurrent test with REAL Fish slow AR (GPU) + REAL fast AR (ANE)
cp -r /tmp/fish_slow_ar_real.mlpackage /tmp/concurrent_large.mlpackage
cp -r /tmp/fish_real_fast_ar.mlpackage /tmp/fish_fast_ar.mlpackage
echo "Running Swift concurrent test with REAL Fish weights (slow AR GPU + fast AR ANE)..."
./concurrent_swift_test
