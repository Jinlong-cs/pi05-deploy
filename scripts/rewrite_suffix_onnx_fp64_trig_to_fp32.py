#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite suffix_step ONNX float64 trig path to float32 for ORT QDQ.")
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--output-onnx", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_json or args.output_onnx.with_suffix(".rewrite_summary.json")

    model = onnx.load(str(args.source_onnx), load_external_data=True)

    patched_cast_nodes: list[str] = []
    patched_constant_nodes: list[str] = []

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                    attr.i = TensorProto.FLOAT
                    patched_cast_nodes.append(node.name)

        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name != "value":
                    continue
                if attr.t.data_type != TensorProto.DOUBLE:
                    continue
                array = numpy_helper.to_array(attr.t).astype("float32")
                new_tensor = numpy_helper.from_array(array, name=attr.t.name or f"{node.name}_fp32")
                attr.CopyFrom(helper.make_attribute("value", new_tensor))
                patched_constant_nodes.append(node.name)

    onnx.save_model(
        model,
        str(args.output_onnx),
        save_as_external_data=False,
    )
    onnx.checker.check_model(onnx.load(str(args.output_onnx), load_external_data=False))

    summary = {
        "source_onnx": str(args.source_onnx),
        "output_onnx": str(args.output_onnx),
        "patched_cast_nodes": patched_cast_nodes,
        "patched_constant_nodes": patched_constant_nodes,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
