from pathlib import Path
import sys
import argparse
import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16

def convert_float16(input_path: str, output_path: str) -> None:
    """
    Convert an ONNX model to FP16 (Float16) precision.

    Args:
        input_path (str): Path to the input ONNX model.
        output_path (str): Path where the FP16 model will be saved.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print("Converting model to FP16...")

        # Load the model
        model = onnx.load(str(input_path))

        # Convert model to FP16
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=True,  # Keep input/output as float32
            disable_shape_infer=True  # Disable shape inference for better compatibility
        )

        # Save the converted model
        onnx.save(model_fp16, str(output_path))

        original_size = input_path.stat().st_size / (1024 * 1024)
        converted_size = output_path.stat().st_size / (1024 * 1024)

        print(f"\nConversion Results:")
        print(f"Model converted and saved to: {output_path}")
        print(f"Original model size: {original_size:.2f} MB")
        print(f"FP16 model size: {converted_size:.2f} MB")
        print(
            f"Size reduction: {((original_size - converted_size) / original_size * 100):.2f}%"
        )

    except PermissionError:
        print(f"Error: Unable to write to {output_path}")
        print("Please check file permissions and try again")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16")
    parser.add_argument(
        "--input", type=str, required=True, help="Input ONNX model path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output FP16 model path"
    )

    args = parser.parse_args()
    convert_float16(args.input, args.output)


if __name__ == "__main__":
    main()
