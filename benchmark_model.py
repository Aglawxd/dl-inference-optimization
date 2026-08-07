import torch
import torchvision.models as models
import time

from sympy.physics.units import acceleration

model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

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

CPU_time = timer(model, example_image)
print(f'\nAverage inference CPU time: {CPU_time:.2f} ms')

if torch.cuda.is_available():
    GPU_model = model.to('cuda')
    GPU_data = example_image.to('cuda')

    GPU_time = timer(GPU_model, GPU_data)
    print(f'\nAverage  inference GPU time: {GPU_time:.2f} ms')

    acceleration = CPU_time / GPU_time
    print(f'\n GPU acceleration time is {acceleration:.1f}x faster than CPU')
else:
    print('GPU not available. Test was not run')