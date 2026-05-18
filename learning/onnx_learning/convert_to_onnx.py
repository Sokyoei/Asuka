"""
convert to .onnx file
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path


class ExportONNX:

    def __init__(self, model_path: str | os.PathLike, input_name: str, output_name: str) -> None:
        self.model_path = Path(model_path)
        self.onnx_name = self.model_path.name
        self.input_name = input_name
        self.output_name = output_name

    def pytorch(self):
        """PyTorch"""
        import torch

        torch.onnx.export(
            torch.load(self.model_path),
            torch.randn(1, 3, 640, 640),
            input_names=self.input_name,
            output_names=self.output_name,
            verbose=True,
        )
        # torch.onnx.dynamo_export()

    def tensorflow(self):
        """TensorFlow"""
        print("tensorflow")
        pass

    def sklearn(self):
        """scikit-learn"""
        print("sklearn")
        pass

    def __getitem__(self, framework: str):
        return {"pytorch": self.pytorch, "tensorflow": self.tensorflow, "sklearn": self.sklearn}.get(framework)


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", help="model path")
    parser.add_argument(
        "-f", "--framework", default="pytorch", help="framework is one of [pytorch, tensorflow, sklearn]"
    )
    parser.add_argument("--input_name", default="input_name", help="input name")
    parser.add_argument("--output_name", default="output_name", help="output name")
    args = parser.parse_args()
    export = ExportONNX(args.model_path, args.input_name, args.output_name)
    export[args.framework]()


if __name__ == "__main__":
    main()
