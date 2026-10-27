from pathlib import Path
import json
import platform
import statistics
import time

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/optimized/dino_disease/onnx/dino_disease_fp32_224.onnx"
REPORT = ROOT / "reports/device_validation/dino_disease/fp32_cpu_thread_benchmark.json"

if not MODEL.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL}")
if REPORT.exists():
    raise FileExistsError(f"Refusing to overwrite report: {REPORT}")

# Fixed synthetic input measures runtime performance, not accuracy.
rng = np.random.default_rng(42)
batch = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

results = []

for threads in [1, 2, 4, 8, 16]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(MODEL),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    # Warm-up
    for _ in range(3):
        session.run(["logits", "embeddings"], {"images": batch})

    timings = []
    for _ in range(15):
        start = time.perf_counter()
        logits, embeddings = session.run(
            ["logits", "embeddings"], {"images": batch}
        )
        timings.append((time.perf_counter() - start) * 1000)

    result = {
        "intra_op_threads": threads,
        "median_latency_ms": round(statistics.median(timings), 3),
        "mean_latency_ms": round(statistics.mean(timings), 3),
        "min_latency_ms": round(min(timings), 3),
        "logits_shape": list(logits.shape),
        "embeddings_shape": list(embeddings.shape),
    }
    results.append(result)
    print(result)

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(
    json.dumps({
        "model": str(MODEL.relative_to(ROOT)),
        "cpu": platform.processor(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "onnxruntime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
        "input_shape": list(batch.shape),
        "warmup_runs": 3,
        "measured_runs_per_configuration": 15,
        "results": results,
    }, indent=2),
    encoding="utf-8",
)

best = min(results, key=lambda item: item["median_latency_ms"])
print("\nFastest tested configuration:", best["intra_op_threads"], "threads")
print("Median latency:", best["median_latency_ms"], "ms")
print("Report:", REPORT)
print("Synthetic-input benchmark only; no accuracy claims.")
