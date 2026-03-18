// Fish S2 Pro ANE Benchmark — exact matmul dimensions
// Build: clang -framework Foundation -framework IOSurface -framework CoreML -lobjc -o fish_ane_bench fish_ane_bench.m
// Run:   ./fish_ane_bench

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static NSData *buildWeightBlob(int out_ch, int in_ch) {
    NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE;
    chunk[4]=0x01; chunk[10]=0x08;
    uint16_t *fp16 = (uint16_t*)(chunk + 64);
    for (NSUInteger j = 0; j < (NSUInteger)out_ch * in_ch; j++)
        fp16[j] = (arc4random() & 0x03FF) | 0x2000;
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *genMIL(int in_ch, int out_ch, int sp) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", in_ch, sp];
    [m appendString:
        @"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        @"        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n", in_ch, sp];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n", out_ch, in_ch, out_ch, in_ch];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n", out_ch, sp];
    [m appendString:@"        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n", out_ch, sp];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

double benchRect(int in_ch, int out_ch, int sp) {
    @autoreleasepool {
        NSError *e = nil;
        NSData *milData = [[genMIL(in_ch, out_ch, sp) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *wb = buildWeightBlob(out_ch, in_ch);

        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

        NSDictionary *wdict = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}
        };
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) return -2;
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return -3;

        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wb writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return -4; }

        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return -5; }

        // IOSurface setup (matching maderix's pattern)
        NSUInteger inBytes = in_ch * sp * 4;
        NSUInteger outBytes = out_ch * sp * 4;

        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(inBytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(inBytes),
            (id)kIOSurfaceAllocSize:@(inBytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(outBytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(outBytes),
            (id)kIOSurfaceAllocSize:@(outBytes),(id)kIOSurfacePixelFormat:@0});

        if (!ioIn || !ioOut) { [fm removeItemAtPath:tmpDir error:nil]; return -6; }

        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        // Warmup
        for (int i = 0; i < 5; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

        // Benchmark
        int iters = 50;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double ms = ticksToMs(mach_absolute_time() - t0) / iters;

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];
        return ms;
    }
}

int main(int argc, char **argv) {
    mach_timebase_info(&g_tb);

    printf("=== Fish S2 Pro ANE Benchmark (M2 Max) ===\n");
    printf("Exact matmul dimensions from Fish S2 Pro transformer\n\n");
    printf("%-35s  W (MB)    ms/eval   GFLOP   TFLOPS\n", "Operation");
    printf("---------------------------------------------------------\n");

    struct { const char *name; int in_ch; int out_ch; int sp; } tests[] = {
        {"Q proj (2560→4096)",           2560, 4096, 1},
        {"K proj (2560→1024)",           2560, 1024, 1},
        {"V proj (2560→1024)",           2560, 1024, 1},
        {"O proj (4096→2560)",           4096, 2560, 1},
        {"FFN gate (2560→9728)",         2560, 9728, 1},
        {"FFN up (2560→9728)",           2560, 9728, 1},
        {"FFN down (9728→2560)",         9728, 2560, 1},
        // Batch mode (seq_len > 1)
        {"Q proj seq=16",                2560, 4096, 16},
        {"FFN gate seq=16",              2560, 9728, 16},
        {"FFN gate seq=64",              2560, 9728, 64},
    };
    int ntests = sizeof(tests)/sizeof(tests[0]);

    for (int i = 0; i < ntests; i++) {
        double wMB = (double)tests[i].in_ch * tests[i].out_ch * 2.0 / (1024.0*1024.0);
        double ms = benchRect(tests[i].in_ch, tests[i].out_ch, tests[i].sp);
        if (ms > 0) {
            double gflop = 2.0 * tests[i].in_ch * tests[i].out_ch * tests[i].sp / 1e9;
            double tflops = gflop / (ms / 1000.0);
            printf("%-35s  %5.1f  %9.3f ms  %5.2f   %5.2f\n", tests[i].name, wMB, ms, gflop, tflops);
        } else {
            printf("%-35s  %5.1f  FAILED (%.0f)\n", tests[i].name, wMB, ms);
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
