import onnx
import onnxsim
import torch
from torch import Tensor, nn

MODEL_NAME = "torch_onnx_example.onnx"
MODEL_SIM_NAME = "torch_onnx_sim_example.onnx"
MODEL_DYNAMIC_SHAPE_NAME = "torch_onnx_dynamic_shape_example.onnx"


class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)
        self.flatten1 = nn.Flatten(2)  # N C H W -> N C H*W
        self.fc1 = nn.Linear(36, 20)
        self.flatten2 = nn.Flatten()  # N C H*W -> N C*H*W
        self.fc2 = nn.Linear(64 * 20, 10)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # self.flatten(x), 但是在 onnx 中生成了 shape->silce->concat->reshape 算子
        # 可以使用 onnx-simplifier 简化
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.flatten2(x)
        x = self.fc2(x)
        return x


model = SimpleNet()


def inference():
    x = torch.randn(1, 3, 24, 24, dtype=torch.float32)
    y = model(x)
    print(f"result: {y=}")


def export():
    x = torch.randn(1, 3, 24, 24)
    model.eval()

    torch.onnx.export(
        model,
        (x,),
        MODEL_NAME,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        # verbose=True,
    )
    print("export done.")

    # onnx-simplifier
    onnx_model = onnx.load(MODEL_NAME)
    onnx.checker.check_model(onnx_model)
    sim_onnx_model, ret = onnxsim.simplify(onnx_model)
    assert ret, "simplifier fail."
    onnx.save(sim_onnx_model, MODEL_SIM_NAME)
    print("simplifier done.")

    # export dynamic shape
    torch.onnx.export(
        model,
        (x,),
        MODEL_DYNAMIC_SHAPE_NAME,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
        # verbose=True,
    )
    print("export dynamic shape done.")

    print("export onnx successfully.")


def main():
    inference()
    export()


if __name__ == "__main__":
    main()
