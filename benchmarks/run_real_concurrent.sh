#!/bin/bash
set -euo pipefail

# Quick concurrent test with real Fish fast AR weights
# Builds concurrent_swift_test on demand unless SWIFT_BIN points elsewhere.

FAST_AR_PACKAGE="${FAST_AR_PACKAGE:-/tmp/fish_real_fast_ar.mlpackage}"
SWIFT_BIN="${SWIFT_BIN:-./concurrent_swift_test}"

if [ ! -x "$SWIFT_BIN" ]; then
    swiftc -framework CoreML -framework Foundation -O concurrent_swift_test.swift -o "$SWIFT_BIN"
fi

# The Swift test looks for /tmp/fish_fast_ar.mlpackage.
cp -r "$FAST_AR_PACKAGE" /tmp/fish_fast_ar.mlpackage

echo "Running Swift concurrent test with REAL Fish fast AR weights..."
"$SWIFT_BIN"
