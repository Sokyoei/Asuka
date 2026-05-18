import onnx
import onnxsim
import torch
import torchvision
from torch import Tensor, nn


def export(pth_model: nn.Module, args: tuple[Tensor]):
    model_name = pth_model._get_name()
    onnx_model_name = f"{model_name}.onnx"

    torch.onnx.export(
        pth_model, args, onnx_model_name, input_names=["input"], output_names=["output"], opset_version=15
    )

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    onnx_model, ret = onnxsim.simplify(onnx_model)
    assert ret, "simplifier fail."
    onnx.save(onnx_model, onnx_model_name)
    print(f"export {model_name} done.")


models = {
    torchvision.models.resnet50(): (torch.randn(1, 3, 224, 224),),
    torchvision.models.vgg19(): (torch.randn(1, 3, 224, 224),),
    torchvision.models.mobilenet_v3_small(): (torch.randn(1, 3, 224, 224),),
    torchvision.models.swin_s(): (torch.randn(1, 3, 224, 224),),
    torchvision.models.alexnet(): (torch.randn(1, 3, 224, 224),),
}


def main():
    for model, args in models.items():
        export(model, args)


if __name__ == "__main__":
    main()
