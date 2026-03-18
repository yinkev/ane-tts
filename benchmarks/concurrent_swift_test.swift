// Swift test: Can CoreML models run concurrently on GPU + ANE via GCD?
//
// This tests the hardware-level concurrency that Python couldn't achieve.
// Uses DispatchQueue for true parallel dispatch, not Python threads.
//
// Build:
//   swiftc -framework CoreML -framework Foundation -O concurrent_swift_test.swift -o concurrent_swift_test
//
// Prerequisites:
//   /tmp/concurrent_large.mlpackage (GPU model) and
//   /tmp/fish_fast_ar.mlpackage (ANE model)
//   must exist from the Python benchmarks.

import Foundation
import CoreML

func benchmark(_ label: String, iterations: Int, block: () -> Void) -> Double {
    // Warmup
    for _ in 0..<3 { block() }

    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations { block() }
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    let ms = elapsed / Double(iterations) * 1000.0
    print("  \(label): \(String(format: "%.3f", ms)) ms/eval")
    return ms
}

print("=== Swift Concurrent GPU + ANE Test ===\n")

// Load models with specific compute units
let gpuConfig = MLModelConfiguration()
gpuConfig.computeUnits = .cpuAndGPU

let aneConfig = MLModelConfiguration()
aneConfig.computeUnits = .all

print("Loading models...")

let gpuURL = URL(fileURLWithPath: "/tmp/concurrent_large.mlpackage")
let aneURL = URL(fileURLWithPath: "/tmp/fish_fast_ar.mlpackage")

// Compile models first
guard let gpuCompiled = try? MLModel.compileModel(at: gpuURL) else {
    print("ERROR: Could not compile GPU model")
    exit(1)
}
guard let aneCompiled = try? MLModel.compileModel(at: aneURL) else {
    print("ERROR: Could not compile ANE model")
    exit(1)
}

guard let gpuModel = try? MLModel(contentsOf: gpuCompiled, configuration: gpuConfig) else {
    print("ERROR: Could not load GPU model")
    exit(1)
}

guard let aneModel = try? MLModel(contentsOf: aneCompiled, configuration: aneConfig) else {
    print("ERROR: Could not load ANE model")
    exit(1)
}

print("  GPU model loaded")
print("  ANE model loaded\n")

// Create input (1, 1, 2560) — matching Fish dimensions
let inputArray = try! MLMultiArray(shape: [1, 1, 2560], dataType: .float32)
for i in 0..<2560 { inputArray[i] = NSNumber(value: Float.random(in: -1...1)) }

let gpuInput = try! MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
let aneInput = try! MLDictionaryFeatureProvider(dictionary: ["input": inputArray])

let iterations = 100

// 1. GPU alone
print("Benchmarking...")
let gpuMs = benchmark("GPU alone", iterations: iterations) {
    let _ = try? gpuModel.prediction(from: gpuInput)
}

// 2. ANE alone
let aneMs = benchmark("ANE alone", iterations: iterations) {
    let _ = try? aneModel.prediction(from: aneInput)
}

// 3. Concurrent via GCD
let gpuQueue = DispatchQueue(label: "gpu.inference", qos: .userInteractive)
let aneQueue = DispatchQueue(label: "ane.inference", qos: .userInteractive)

// Warmup concurrent
for _ in 0..<3 {
    let group = DispatchGroup()
    group.enter(); gpuQueue.async { let _ = try? gpuModel.prediction(from: gpuInput); group.leave() }
    group.enter(); aneQueue.async { let _ = try? aneModel.prediction(from: aneInput); group.leave() }
    group.wait()
}

let concurrentStart = CFAbsoluteTimeGetCurrent()
for _ in 0..<iterations {
    let group = DispatchGroup()
    group.enter()
    gpuQueue.async {
        let _ = try? gpuModel.prediction(from: gpuInput)
        group.leave()
    }
    group.enter()
    aneQueue.async {
        let _ = try? aneModel.prediction(from: aneInput)
        group.leave()
    }
    group.wait()
}
let concurrentMs = (CFAbsoluteTimeGetCurrent() - concurrentStart) / Double(iterations) * 1000.0
print("  Concurrent (GCD): \(String(format: "%.3f", concurrentMs)) ms/eval")

// Analysis
let sequentialMs = gpuMs + aneMs
let parallelIdeal = max(gpuMs, aneMs)

print("\n=== Results ===")
print("  GPU alone:        \(String(format: "%.3f", gpuMs)) ms")
print("  ANE alone:        \(String(format: "%.3f", aneMs)) ms")
print("  Sequential sum:   \(String(format: "%.3f", sequentialMs)) ms")
print("  Parallel ideal:   \(String(format: "%.3f", parallelIdeal)) ms")
print("  Concurrent actual:\(String(format: "%.3f", concurrentMs)) ms")

let overlap = (sequentialMs - concurrentMs) / (sequentialMs - parallelIdeal) * 100.0
print("\n  Overlap: \(String(format: "%.0f", overlap))%")

if overlap > 60 {
    print("  GPU and ANE run in PARALLEL via Swift GCD!")
    print("  Pipeline parallelism for Fish S2 Pro is CONFIRMED.")
    let fishSpeedup = 66.7 / max(34.7, concurrentMs * 34.7 / gpuMs)
    print("  Expected Fish speedup: \(String(format: "%.2f", fishSpeedup))x")
} else if overlap > 30 {
    print("  Partial overlap — some benefit but not full parallelism.")
} else {
    print("  Minimal overlap — CoreML may serialize even in Swift.")
}

print("\n=== Done ===")
