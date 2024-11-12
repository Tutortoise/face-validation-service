from onnxruntime.quantization import quantize_dynamic, QuantType, create_calibrator
from pathlib import Path
import sys
import argparse
import onnx
from onnx import version_converter


def update_model_opset(model_path: str, target_opset: int) -> str:
    """
    Update the ONNX model to a higher opset version.

    Args:
        model_path (str): Path to the input ONNX model
        target_opset (int): Target ONNX opset version

    Returns:
        str: Path to the updated model
    """
    print(f"Updating model to ONNX opset version {target_opset}...")
    model = onnx.load(model_path)

    converted_model = version_converter.convert_version(model, target_opset)

    updated_path = str(Path(model_path).parent / f"{Path(model_path).stem}_opset{target_opset}.onnx")
    onnx.save(converted_model, updated_path)
    print(f"Updated model saved to: {updated_path}")
    return updated_path


def quantize_onnx_model(input_path: str, output_path: str) -> None:
    """
    Quantize an ONNX model using dynamic quantization.

    Args:
        input_path (str): Path to the input ONNX model.
        output_path (str): Path where the quantized model will be saved.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        updated_model_path = update_model_opset(str(input_path), 19)

        print("Quantizing YOLO model...")

        # Configure quantization parameters
        quantization_params = {
            "model_input": updated_model_path,
            "model_output": str(output_path),
            "weight_type": QuantType.QUInt8,
            "per_channel": False,
            "reduce_range": False,
            "op_types_to_quantize": ["Conv", "MatMul", "Gemm", "Add", "Mul"],
            "extra_options": {
                "WeightSymmetric": False,
                "EnableSubgraph": True,
                "ForceQuantize": False,
            }
        }

        quantize_dynamic(**quantization_params)

        original_size = input_path.stat().st_size / (1024 * 1024)
        quantized_size = output_path.stat().st_size / (1024 * 1024)

        print(f"\nQuantization Results:")
        print(f"Model quantized and saved to: {output_path}")
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(
            f"Size reduction: {((original_size - quantized_size) / original_size * 100):.2f}%"
        )

        Path(updated_model_path).unlink()

    except PermissionError:
        print(f"Error: Unable to write to {output_path}")
        print("Please check file permissions and try again")
        sys.exit(1)
    except Exception as e:
        print(f"Error during quantization: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model")
    parser.add_argument(
        "--input", type=str, required=True, help="Input ONNX model path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output quantized model path"
    )

    args = parser.parse_args()
    quantize_onnx_model(args.input, args.output)


if __name__ == "__main__":
    main()
