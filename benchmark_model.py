import torch
import torchvision.models as models
import time
from torch.profiler import profile, ProfilerActivity

model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()
CPU_model = models.resnet18(weights= 'IMAGENET1K_V1')
CPU_model.eval()

print('Model loaded')
print('Parameters number: ' , sum(p.numel() for p in model.parameters()))

example_image = torch.randn(32, 3, 224, 224) #ammount of new images added to program

def timer(model, data, reps= 100):
    with torch.no_grad():
        for _ in range(10):
            model(data)
    start = time.time()
    with torch.no_grad():
        for _ in range(reps):
            model(data)
    end = time.time()

    avg_time_ms = (end - start) / reps * 1000
    return avg_time_ms

CPU_time = timer(CPU_model, example_image)
print(f'\nAverage inference CPU time: {CPU_time:.2f} ms')

if torch.cuda.is_available():
    GPU_model = models.resnet18(weights= 'IMAGENET1K_V1')
    GPU_model.eval()
    GPU_model = GPU_model.to('cuda')#varaible moved from CPU to GPU
    GPU_example_image = example_image.to('cuda') #varaible moved from CPU to GPU

    GPU_time = timer(GPU_model, GPU_example_image)
    print(f'\nAverage  inference GPU time: {GPU_time:.2f} ms')

    acceleration = CPU_time / GPU_time
    print(f'\n GPU acceleration time is {acceleration:.1f}x faster than CPU')
else:
    print('GPU not available. Test was not run')


print('\n--- CPU Profiling ---')
with profile(activities = [ProfilerActivity.CPU], record_shapes=True) as prof_CPU:
    with torch.no_grad():
        CPU_model(example_image)

print(prof_CPU.key_averages().table(sort_by='cpu_time_total', row_limit=10))

if torch.cuda.is_available():
    print('\n --- GPU Profiling ---')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True) as prof_GPU:
        with torch.no_grad():
            GPU_model(GPU_example_image)

    print(prof_GPU.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))

def count_GPU_time(model, data, reps= 100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(10):
            model(data)
    torch.cuda.synchronize()

    start_event.record()
    with torch.no_grad():
        for _ in range(reps):
            model(data)
    end_event.record()

    torch.cuda.synchronize()
    full_time_ms = start_event.elapsed_time(end_event)
    return full_time_ms / reps

if torch.cuda.is_available():
    GPU_time_precise = count_GPU_time(GPU_model, GPU_example_image )
    print(f'\nPrecise average GPU time: (torch.cuda.Event: {GPU_time_precise:.2f}')

print('\n --- Optimaize: torch.compile() --- ')
model_compiled = models.resnet18(weights = 'IMAGENET1K_V1')
model_compiled.eval()
model_compiled = model_compiled.to('cuda')
model_compiled = torch.compile(model_compiled)

print('Model compiling. Please wait')
with torch.no_grad():
    for _ in range(10):
        model_compiled(GPU_example_image)
torch.cuda.synchronize()
print('Model compiled')

time_compiled = count_GPU_time(model_compiled, GPU_example_image)
print(f'Average GPU time after torch.compile(): {time_compiled:.2f} ms')

acceleration_compile = GPU_time_precise / time_compiled
print(f'torch.compile() gave {acceleration_compile:.2f}x change comapred to GPU without optimize')

print('\n --- Optimaize: fp16 (half precision) ---')

model_fp16 = models.resnet18(weights = 'IMAGENET1K_V1')
model_fp16.eval()
model_fp16 = model_fp16.to('cuda').half()

data_fp16 = GPU_example_image.to('cuda').half()

time_fp16 = count_GPU_time(model_fp16, data_fp16)
print(f'Average fp16 GPU time: {time_fp16:.2f} ms')

acceleration_fp16 = GPU_time_precise / time_fp16
print(f'fp16 gave {acceleration_fp16:.2f}x change compared to default fp32')

print('\n --- Optimize: torch.acutocast (mixed precison) --- ')
model_autocast = models.resnet18(weights = 'IMAGENET1K_V1')
model_autocast.eval()
model_autocast = model_autocast.to('cuda')

def count_time_autocast(model, data, reps=100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(10):
                model(data)
    torch.cuda.synchronize()

    start_event.record()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(reps):
                model(data)
    end_event.record()

    torch.cuda.synchronize()
    full_time_ms = start_event.elapsed_time(end_event)
    return full_time_ms / reps

time_autocast = count_time_autocast(model_autocast, GPU_example_image)
print(f'Average GPU time with autocast: {time_autocast:.2f} ms')

acceleration_autocast = GPU_time_precise / time_autocast
print(f'Autocast gave {acceleration_autocast:.2f}x change compred to GPU without optimize')

print('\n --- Optimaize: cudnn.benchmark ----')

torch.backends.cudnn.benchmark = True

model_benchmark = models.resnet18(weights = 'IMAGENET1K_V1')
model_benchmark.eval()
model_benchmark = model_benchmark.to('cuda')

time_benchmark = count_GPU_time(model_benchmark, GPU_example_image)
print(f'Average GPU time with cudnn.benchmark: {time_benchmark:.2f} ms')

acceleration_benchmark = GPU_time_precise / time_benchmark
print(f'cudnn.benchmark gave {acceleration_benchmark:.2f}x change compred to GPU without optimize')

print('\n ---Test with bigger data ')