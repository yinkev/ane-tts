#!/bin/bash
# Quick concurrent test with real Fish fast AR weights
# Uses the already-built concurrent_swift_test but with the real model

# The Swift test looks for /tmp/fish_fast_ar.mlpackage — copy our real one there
cp -r /tmp/fish_real_fast_ar.mlpackage /tmp/fish_fast_ar.mlpackage 2>/dev/null

echo "Running Swift concurrent test with REAL Fish fast AR weights..."
./concurrent_swift_test
