from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from pathlib import Path
import sys
import argparse


def preprocess_model(input_path: str, output_path: str) -> None:
    """
    Preprocess the ONNX model to optimize it for quantization.

    Args:
        input_path (str): Path to the input ONNX model.
        output_path (str): Path where the preprocessed model will be saved.
    """
    try:
        print("Preprocessing the model for quantization...")
        quant_pre_process(input_path, output_path)
        print(f"Preprocessed model saved to: {output_path}")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        sys.exit(1)


def quantize_onnx_model(
    input_path: str, output_path: str, skip_preprocess: bool = False
) -> None:
    """
    Quantize an ONNX model using dynamic quantization.

    Args:
        input_path (str): Path to the input ONNX model.
        output_path (str): Path where the quantized model will be saved.
        skip_preprocess (bool): Skip preprocessing steps if True.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not skip_preprocess:
            prep_path = (
                input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}"
            )
            preprocess_model(str(input_path), str(prep_path))
            model_to_quantize = prep_path
        else:
            model_to_quantize = input_path

        print("Quantizing YOLO model...")
        quantize_dynamic(
            model_input=str(model_to_quantize),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "MatMul"],
            reduce_range=True,
        )

        # Calculate and display size reduction
        original_size = input_path.stat().st_size / (1024 * 1024)
        quantized_size = output_path.stat().st_size / (1024 * 1024)

        print(f"\nQuantization Results:")
        print(f"Model quantized and saved to: {output_path}")
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(
            f"Size reduction: {((original_size - quantized_size) / original_size * 100):.2f}%"
        )

        # Cleanup preprocessed model if it exists
        if not skip_preprocess and prep_path.exists():
            prep_path.unlink()

    except PermissionError:
        print(f"Error: Unable to write to {output_path}")
        print("Please check file permissions and try again")
        sys.exit(1)
    except Exception as e:
        print(f"Error during quantization: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Preprocess and quantize ONNX model")
    parser.add_argument(
        "--input", type=str, required=True, help="Input ONNX model path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output quantized model path"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing steps (not recommended)",
    )

    args = parser.parse_args()
    quantize_onnx_model(args.input, args.output, args.skip_preprocess)


if __name__ == "__main__":
    main()
