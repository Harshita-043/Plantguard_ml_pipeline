from pathlib import Path
import json

import onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)

SOURCE = Path(
    "artifacts/optimized/dino_disease/onnx/"
    "dino_disease_fp32_224.onnx"
)

OUTPUT = Path(
    "artifacts/optimized/dino_disease/onnx_selective_int8/"
    "dino_disease_mlp_int8.onnx"
)

REPORT = Path(
    "reports/device_validation/dino_disease/"
    "selective_int8_quantization.json"
)

if not SOURCE.is_file():
    raise FileNotFoundError(f"FP32 source model not found: {SOURCE}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

model = onnx.load(str(SOURCE), load_external_data=False)

# Quantize only MLP projections and the final classification head.
# Leave all attention operations and the embedding output path
# unquantized unless they are structurally part of an MLP node.
selected_nodes = [
    node.name
    for node in model.graph.node
    if node.op_type in {"MatMul", "Gemm"}
    and (
        "/mlp/fc1/" in node.name
        or "/mlp/fc2/" in node.name
        or node.name == "/head/head.1/Gemm"
    )
]

attention_nodes = [
    node.name
    for node in model.graph.node
    if node.op_type in {"MatMul", "Gemm"}
    and "/attn/" in node.name
]

print(f"Source model: {SOURCE}")
print(f"Matrix operations selected for INT8: {len(selected_nodes)}")
print(f"Attention operations retained in FP32: {len(attention_nodes)}")

for name in selected_nodes:
    print("  INT8:", name)

if not selected_nodes:
    raise RuntimeError("No matching MLP/head nodes found; refusing to quantize.")

if OUTPUT.resolve() == SOURCE.resolve():
    raise RuntimeError("Refusing to overwrite the FP32 source model.")

quantize_dynamic(
    model_input=str(SOURCE),
    model_output=str(OUTPUT),
    nodes_to_quantize=selected_nodes,
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["MatMul", "Gemm"],
)

result = onnx.load(str(OUTPUT), load_external_data=False)

report = {
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "quantization": "dynamic QInt8, per-channel weights",
    "selected_nodes": selected_nodes,
    "attention_nodes_excluded_by_selection": attention_nodes,
    "source_bytes": SOURCE.stat().st_size,
    "output_bytes": OUTPUT.stat().st_size,
    "size_reduction_percent": round(
        100 * (1 - OUTPUT.stat().st_size / SOURCE.stat().st_size), 3
    ),
    "graph_outputs": [item.name for item in result.graph.output],
}

REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nSelective quantization complete.")
print(f"Output: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / (1024**2):.2f} MiB")
print(f"Size reduction: {report['size_reduction_percent']}%")
print(f"Report: {REPORT}")
print("Graph outputs:", report["graph_outputs"])
