# Project Report: Deep Learning Inference Optimization

**Repository:** `dl-inference-optimization`
**Test hardware:** NVIDIA GeForce RTX 2060 (CUDA 13.1 driver), PyTorch 2.13.0+cu130
**Reference model:** ResNet18 (torchvision, IMAGENET1K_V1 weights, 11,689,512 parameters)

---

## 1. Project Goal

This project reproduces, at a smaller scale, three tasks typical of AI performance engineering in a production environment:

1. **Measuring and improving inference speed** on GPU accelerators.
2. **Profiling DL workloads** to identify bottlenecks.
3. **Deploying a model as a production service** (REST API) with realistic traffic patterns in mind.

The reference model (ResNet18) serves as a test workload — the goal is not classification quality, but performance characterization and methodology, transferable to other architectures.

---

## 2. Measurement Methodology

Two timing methods were used, deliberately compared against each other:

| Method | Use case | Notes |
|---|---|---|
| `time.time()` | CPU, orientative measurement | Unreliable on GPU due to asynchronous execution |
| `torch.cuda.Event` + `torch.cuda.synchronize()` | GPU, precise measurement | Guarantees the reading happens after computation actually finishes on the device |

Every measurement was preceded by a **warm-up phase** (10 calls excluded from statistics) — the first call to a model is systematically slower (memory allocation, CUDA kernel initialization) and would distort the average. Each actual measurement is an average of 100 repetitions.

Cross-validation of both methods on the same workload produced consistent results (4.25–4.67 ms), confirming the reliability of the simpler method where the precise one was not yet implemented.

---

## 3. Results: CPU vs GPU Baseline

| Batch size | CPU (ms) | GPU (ms) | Speedup |
|---|---|---|---|
| 1 | 38.86–39.72 | 4.25–4.67 | ~9.1× |
| 32 | 1011.46 | 25.19–25.70 | ~40.2× |

**Observation:** the GPU advantage grows non-linearly with batch size — CPU scales nearly linearly with the number of images, while GPU leverages hardware parallelism and grows much more slowly. This is an early signal that a GPU of this class is underutilized on single requests.

---

## 4. Profiling (`torch.profiler`)

### CPU — execution time breakdown

| Operation | Share of total time |
|---|---|
| `aten::mkldnn_convolution` | 81–85% |
| `aten::max_pool2d_with_indices` | 6–9% |
| `aten::batch_norm` (and derivatives) | 4–5% |
| remaining | <2% |

Convolutions clearly dominate — any CPU-side optimization should focus exclusively on this operation.

### GPU — execution time breakdown

| Operation | Share of total time |
|---|---|
| `aten::cudnn_convolution` | 25–35% |
| `aten::cudnn_batch_norm` | 22–26% |
| management overhead (`empty`, `view`, `empty_like`) | ~10–15% |

**Conclusion:** on GPU, the distribution is significantly more even. Convolutions stop dominating (cuDNN executes them very efficiently), while the proportional share of batch normalization and memory-management overhead increases. This means GPU-side optimization requires a different approach than CPU-side — speeding up a single operation is not enough.

---

## 5. Optimization Technique Testing

Four standard inference optimization techniques were tested, each measured against the same baseline (`torch.cuda.Event`, fp32, no optimization).

### At batch_size = 1

| Technique | Result | Change vs. baseline |
|---|---|---|
| `torch.compile()` | 5.06–5.19 ms | 0.86–0.95× (**slower**) |
| `.half()` (manual fp16) | 6.25–6.44 ms | 0.72× (**slower**) |
| `torch.autocast` (mixed precision) | 5.70–5.82 ms | 0.78–0.80× (**slower**) |
| `torch.backends.cudnn.benchmark` | 4.59 ms | 0.99× (no change) |

**None of the four techniques produced an improvement.** This result, while counterintuitive relative to documentation claims for these tools, is fully explainable (see Section 6).

### At batch_size = 32

| Technique | Result | Change vs. baseline |
|---|---|---|
| `torch.compile()` | — | 0.79× (still slower) |
| `.half()` (manual fp16) | — | **1.99×** (faster) |
| `torch.autocast` (mixed precision) | — | **2.01×** (faster) |
| `torch.backends.cudnn.benchmark` | — | 1.03× (marginal) |

**Key project finding:** fp16 and mixed precision, useless on single requests, deliver nearly a 2x speedup under larger parallel workloads.

---

## 6. Root Cause Analysis

Three factors explain the results above:

1. **GPU underutilization at batch=1.** A single request does not generate enough parallel operations to offset the overhead of precision management (fp32↔fp16 conversion) or graph compilation (`torch.compile`). Administrative overhead outweighs the computational gain.

2. **GPU architecture generation.** The RTX 2060 (Turing, 2018) has first-generation Tensor Cores, with significantly lower throughput than Ampere/Ada/Hopper chips, where the largest mixed-precision gains are typically documented. The profiler message `Not enough SMs to use max_autotune_gemm mode` directly confirms a hardware limitation relative to `torch.compile`'s aggressive optimization modes.

3. **Model scale.** ResNet18 (11.7M parameters) is already close to optimally handled by native cuDNN in fp32. Optimization techniques reveal their value more strongly on larger, deeper architectures, where administrative cost is relatively smaller compared to the volume of computation.

**Methodological conclusion:** the effectiveness of inference optimization techniques is not a universal property — it depends jointly on hardware, model size, and the actual workload pattern. Optimization decisions require measurement under conditions close to production, not extrapolation from documentation or benchmark results published on different hardware.

---

## 7. Production Deployment: Dynamic Batching API

### Design Problem

The result from Section 5 creates a practical conflict: a typical API request handles a single image (batch=1) — exactly the scenario in which fp16/autocast provide no benefit. Building an API without accounting for this would result in an architecture that never realizes the measured speedup potential.

### Solution: Dynamic Batching

A pattern used in production inference servers was implemented (conceptually similar to Triton Inference Server / TorchServe):

```
Client requests (asynchronous, individual)
        │
        ▼
   queue.Queue()  ──►  background thread (daemon)
        │                    │
        │        collects requests over a time window
        │        (max 50 ms or up to MAX_BATCH_SIZE=32)
        │                    │
        │        stacks into a single batch tensor
        │        runs inference under torch.autocast (fp16)
        │                    │
        │        splits results back to individual
        │        requests (threading.Condition)
        ▼                    ▼
   response 1  ◄──────  result for request 1
   response 2  ◄──────  result for request 2
   ...
```

**Implementation elements:**
- `queue.Queue` — thread-safe queue accepting incoming requests
- a `daemon` thread running independently of Flask's request–response cycle
- `threading.Condition` — synchronization without active polling (busy-waiting)
- `app.run(threaded=True)` — handling multiple concurrent client connections

### Load Test Result

Test: 10 concurrent requests (separate client threads, fired at the same moment).

| Metric | Value |
|---|---|
| Requests collected into a single batch | 10 / 10 (100%) |
| Full batch inference time | 268.68 ms |
| Per-image processing time within batch | ~26.9 ms |
| Total response time (round-trip) per request | 346–375 ms |

**Interpretation:** all concurrent requests were correctly merged into a single batch and processed together using `torch.autocast`, realizing the speedup measured in Section 5. The gap between pure inference time (268 ms) and full round-trip (~355 ms) is due to waiting time within the batching window plus HTTP communication overhead — this is a deliberate trade-off: **increased system throughput at the cost of individual request latency**, typical of batching architectures in production.

---

## 8. Project Limitations

- Tests were run on a single consumer-grade GPU (RTX 2060); quantitative results do not directly extrapolate to data-center-class cards (A100, H100).
- The batching window (50 ms) and maximum batch size (32) were chosen empirically for demonstration purposes; production tuning would require analysis of actual traffic patterns and an acceptable latency SLA.
- `torch.compile()` was not subjected to deeper root-cause diagnostics (e.g., inspection of generated Triton code) — a potential direction for further work.
- The API test uses synthetic input data (random tensors) — a production environment would require validation with real images and handling of malformed/invalid input.

---

## 9. Conclusions

1. GPU delivers a 9–40x speedup relative to CPU for this workload, with the advantage growing with batch size.
2. Standard inference optimization techniques (mixed precision, `torch.compile`, cuDNN autotuning) **do not work universally** — their effectiveness depends on batch size, GPU architecture, and model scale, and requires empirical verification under conditions close to the target deployment.
3. With appropriately sized workloads (batch=32), mixed precision (fp16/autocast) delivered a consistent, nearly 2x speedup.
4. Recognizing the gap between single-request and batched-request performance characteristics made it possible to design a deployment architecture (dynamic batching) that realizes the measured optimization potential even under typical, single-request API traffic — without this mechanism, the fp16 gain would have remained purely theoretical.
