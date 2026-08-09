from flask import Flask, request, jsonify
import torch
import torchvision
import torchvision.models as models
import threading
import queue
import time
import uuid
app = Flask(__name__)

model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()
model = model.to('cuda')
print('Model loaded and moved to GPU')

BATCH_WINDOW_MS = 50
MAX_BATCH_SIZE = 32
request_queue = queue.Queue()
results = {}
results_lock = threading.Lock()
new_result_event = threading.Condition()

def batching_worker():
    '''Background thread: continuously collects incoming requests
    into bathes and runs inference, then makes results available'''
    while True:
        batch_items = []

        first_item = request_queue.get()
        batch_items.append(first_item)

        window_start = time.time()
        while len(batch_items) < BATCH_WINDOW_MS:
            elapsed_ms = (time.time() - window_start) * 1000
            remaining_ms = BATCH_WINDOW_MS - elapsed_ms
            if remaining_ms <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining_ms / 1000)
                batch_items.append(item)
            except queue.Empty:
                break

        batch_tensor = torch.cat([item['tensor'] for item in batch_items ], dim=0)
        batch_tensor = batch_tensor.to('cuda')

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                start = time.time()
                predictions = model(batch_tensor)
                torch.cuda.synchronize()
                inference_time_ms = (time.time() - start) * 1000

        with results_lock:
            for i, item in enumerate(batch_items):
                predicted_class = predictions[i].argmax().item()
                results[item['id']] = {
                    'predicted_class': predicted_class,
                    'batch_size': len(batch_items),
                    'batch_inference_ms': round(inference_time_ms, 2),
                }

        with new_result_event:
            new_result_event.notify_all()

worker_thread = threading.Thread(target=batching_worker, daemon=True)
worker_thread.start()

@app.route('/predict', methods=['POST'])
def predict():

    image_tensor = torch.randn(1,3,224,224)

    request_id = str(uuid.uuid4())
    request_queue.put({'id': request_id, 'tensor': image_tensor})

    with new_result_event:
        while request_id not in results:
            new_result_event.wait()
    with results_lock:
        result = results.pop(request_id)

    return jsonify(result)

@app.route('/', methods= ['GET'])
def home():
    return jsonify({'Status': "Dynamic batching API running"})

if __name__ == '__main__':
    app.run(debug=False, threaded = True, port = 5000)
