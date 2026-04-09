"""
TensorRT Utils
"""

import os
from abc import ABC

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
from numpy.typing import NDArray


def _get_tensorrt_version():
    """获取 TensorRT 版本号"""
    return tuple(int(i) for i in trt.__version__.split(".")[:3])


TENSORRT_MAJOR, TENSORRT_MINOR, TENSORRT_PATCH = _get_tensorrt_version()


class TensorRTModel(ABC):

    def __init__(self, model_path: str | os.PathLike, in_thread: bool = False):
        """初始化并加载 TensorRT 模型"""
        self.in_thread = in_thread
        if self.in_thread:
            self.cfx = cuda.Device(0).make_context()  # set is for inference in thread
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(model_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()  # 创建 CUDA 流

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        if TENSORRT_MAJOR >= 10:
            self.input_shape = self.engine.get_tensor_shape(self.input_name)
            self.output_shape = self.engine.get_tensor_shape(self.output_name)
        else:
            self.input_idx = self.engine.get_binding_index(self.input_name)
            self.output_idx = self.engine.get_binding_index(self.output_name)
            self.input_shape = self.engine.get_binding_shape(self.input_idx)
            self.output_shape = self.engine.get_binding_shape(self.output_idx)

        # 分配 GPU 内存
        self.input_memory = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
        self.output_memory = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)
        self.bindings = [int(self.input_memory), int(self.output_memory)]

    def load_engine(self, model_path: str | os.PathLike):
        """加载 TensorRT 引擎模型"""
        with open(model_path, "rb") as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def inference(self, input_data: NDArray) -> NDArray:
        """执行推理"""
        # 确保输入数据的大小匹配
        assert input_data.shape == tuple(
            self.input_shape
        ), f"输入数据的形状应为 {self.input_shape}, 但得到 {input_data.shape}"

        if self.in_thread:
            self.cfx.push()

        if TENSORRT_MAJOR >= 10:
            self.context.set_input_shape(self.input_name, (1, 3, input_data.shape[2], input_data.shape[3]))
            # 设置输入和输出张量地址
            self.context.set_tensor_address(self.input_name, int(self.input_memory))
            self.context.set_tensor_address(self.output_name, int(self.output_memory))
        else:
            self.context.set_binding_shape(0, (1, 3, input_data.shape[2], input_data.shape[3]))

        # 将输入数据传输到 GPU
        cuda.memcpy_htod_async(self.input_memory, input_data, self.stream)

        # 执行推理
        if TENSORRT_MAJOR >= 10:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 创建输出数组并从 GPU 复制到 CPU
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.output_memory, self.stream)

        # 同步 CUDA 流
        self.stream.synchronize()

        if self.in_thread:
            self.cfx.pop()

        return output_data

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError
