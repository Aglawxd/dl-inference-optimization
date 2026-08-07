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

example_image = torch.randn(1, 3, 224, 224)

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
    GPU_model = GPU_model.to('cuda')
    GPU_example_image = example_image.to('cuda')

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

