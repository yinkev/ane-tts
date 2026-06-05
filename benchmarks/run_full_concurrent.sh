#!/bin/bash
set -euo pipefail

# Concurrent test with REAL Fish slow AR (GPU) + REAL fast AR (ANE)
SLOW_AR_PACKAGE="${SLOW_AR_PACKAGE:-/tmp/fish_slow_ar_real.mlpackage}"
FAST_AR_PACKAGE="${FAST_AR_PACKAGE:-/tmp/fish_real_fast_ar.mlpackage}"
SWIFT_BIN="${SWIFT_BIN:-./concurrent_swift_test}"

if [ ! -x "$SWIFT_BIN" ]; then
    swiftc -framework CoreML -framework Foundation -O concurrent_swift_test.swift -o "$SWIFT_BIN"
fi

cp -r "$SLOW_AR_PACKAGE" /tmp/concurrent_large.mlpackage
cp -r "$FAST_AR_PACKAGE" /tmp/fish_fast_ar.mlpackage
echo "Running Swift concurrent test with REAL Fish weights (slow AR GPU + fast AR ANE)..."
"$SWIFT_BIN"
