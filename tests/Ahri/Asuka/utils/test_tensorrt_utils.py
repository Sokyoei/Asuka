import numpy as np

from asuka import ASUKA_ROOT
from asuka.utils.tensorrt_utils import TensorRTModel


def main():
    # 加载并初始化模型
    model_path = str(ASUKA_ROOT / "models/your_model.trtmodel")  # 替换为您的 .trtmodel 路径
    model = TensorRTModel(model_path)

    # 生成随机输入数据进行测试 (根据模型的输入形状调整)
    input_data = np.random.random(model.input_shape).astype(np.float32)  # noqa: NPY002

    # 进行推理并获取结果
    output_data = model.inference(input_data)

    # 打印推理结果
    print("Inference output:", output_data)


if __name__ == "__main__":
    main()
