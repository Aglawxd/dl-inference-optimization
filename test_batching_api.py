import requests
import threading
import time

def send_request(request_number):
    start = time.time()
    response = requests.post('http://127.0.0.1:5000/predict')
    elapsed = (time.time() - start) * 1000
    print(f'Request {request_number}: {response.json()} (total round-trip: {elapsed:.1f} ms')

threads = []
for i in range(10):
    t = threading.Thread(target = send_request, args = (i,))
    threads.append(t)
    t.start()

for i in threads:
    t.join()
