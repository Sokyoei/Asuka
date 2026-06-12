# ruff: noqa: E402

"""
fork https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/common
"""

import gi

gi.require_version('Gst', '1.0')

import ctypes
import platform
import sys
import time
from pathlib import Path
from threading import Lock

from cuda.bindings import driver, runtime
from gi.repository import GLib, Gst
from loguru import logger

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

guard_platform_info = Lock()


def bus_call(bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.error("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.error(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error(f"Error: {err}: {debug}")
        loop.quit()
    return True


class GETFPS:
    def __init__(self, stream_id):
        self.start_time = None
        self.is_first = True
        self.frame_count = 0
        self.stream_id = stream_id
        self.lock = Lock()

    def update_fps(self):
        now = time.time()
        with self.lock:
            if self.is_first:
                self.start_time = now
                self.is_first = False
            else:
                self.frame_count += 1

    def get_fps(self):
        now = time.time()
        with self.lock:
            stream_fps = 0.0 if self.frame_count == 0 else float(self.frame_count / (now - self.start_time))
            self.frame_count = 0
            self.start_time = now
        return round(stream_fps, 2)

    def print_data(self):
        with self.lock:
            logger.info('frame_count=', self.frame_count)
            logger.info('start_time=', self.start_time)


class PERF_DATA:
    def __init__(self, streams: list[tuple[str, str]]):
        self.streams = streams
        self.perf_dict = {}
        self.all_stream_fps: dict[str, GETFPS] = {}
        for name, _ in streams:
            self.all_stream_fps[str(name)] = GETFPS(str(name))

    def perf_print_callback(self):
        self.perf_dict = {stream_index: stream.get_fps() for (stream_index, stream) in self.all_stream_fps.items()}
        logger.info(f"[PERF]: {self.perf_dict}")
        return True

    def update_fps(self, stream_name: str):
        self.all_stream_fps[str(stream_name)].update_fps()


class PlatformInfo:
    def __init__(self):
        self.is_wsl_system = False
        self.wsl_verified = False
        self.is_integrated_gpu_system = False
        self.is_integrated_gpu_verified = False
        self.is_aarch64_platform = False
        self.is_aarch64_verified = False
        self.is_dgx_spark_platform = False

    def is_wsl(self):
        with guard_platform_info:
            # Check if its already verified as WSL system or not.
            if not self.wsl_verified:
                try:
                    # Open /proc/version file
                    with open("/proc/version") as version_file:
                        # Read the content
                        version_info = version_file.readline()
                        version_info = version_info.lower()
                        self.wsl_verified = True

                        # Check if "microsoft" is present in the version information
                        if "microsoft" in version_info:
                            self.is_wsl_system = True
                except Exception as e:
                    logger.error(f"ERROR: Opening /proc/version failed: {e}")

        return self.is_wsl_system

    def is_integrated_gpu(self):
        # Using cuda apis to identify whether integrated/discreet
        # This is required to distinguish Tegra and ARM_SBSA devices
        with guard_platform_info:
            # Cuda initialize
            if not self.is_integrated_gpu_verified:
                (cuda_init_result,) = driver.cuInit(0)
                if cuda_init_result == driver.CUresult.CUDA_SUCCESS:
                    # Get cuda devices count
                    device_count_result, num_devices = driver.cuDeviceGetCount()
                    if device_count_result == driver.CUresult.CUDA_SUCCESS:
                        # If atleast one device is found, we can use the property from
                        # the first device
                        if num_devices >= 1:
                            # Get properties from first device
                            property_result, properties = runtime.cudaGetDeviceProperties(0)
                            if property_result == runtime.cudaError_t.cudaSuccess:
                                logger.info("Is it Integrated GPU? :", properties.integrated)
                                self.is_integrated_gpu_system = properties.integrated
                                self.is_integrated_gpu_verified = True
                            else:
                                logger.error(f"ERROR: Getting cuda device property failed: {property_result}")
                        else:
                            logger.error("ERROR: No cuda devices found to check whether iGPU/dGPU")
                    else:
                        logger.error(f"ERROR: Getting cuda device count failed: {device_count_result}")
                else:
                    logger.error(f"ERROR: Cuda init failed: {cuda_init_result}")

        return self.is_integrated_gpu_system

    def is_platform_aarch64(self):
        # Check if platform is aarch64 using uname
        if not self.is_aarch64_verified:
            if platform.uname()[4] == 'aarch64':
                self.is_aarch64_platform = True
            self.is_aarch64_verified = True
        return self.is_aarch64_platform

    DMI_PATHS = {
        "product_name": Path("/sys/class/dmi/id/product_name"),
        "board_name": Path("/sys/class/dmi/id/board_name"),
        "product_sku": Path("/sys/class/dmi/id/product_sku"),
        "sys_vendor": Path("/sys/class/dmi/id/sys_vendor"),
    }

    def read_dmi_field(self, path: Path) -> str:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            pass
        return ""

    def is_dgx_spark(self) -> bool:
        """
        Return True if this system is detected as DGX Spark, else False.
        Detection is based on DMI product/board/SKU strings.
        """
        product_name = self.read_dmi_field(self.DMI_PATHS["product_name"])
        board_name = self.read_dmi_field(self.DMI_PATHS["board_name"])
        product_sku = self.read_dmi_field(self.DMI_PATHS["product_sku"])

        combined = " ".join(s for s in (product_name, board_name, product_sku) if s).lower()

        self.is_dgx_spark_platform = "dgx spark" in combined
        return self.is_dgx_spark_platform


def long_to_uint64(l):  # noqa: E741
    value = ctypes.c_uint64(l & 0xFFFFFFFFFFFFFFFF).value
    return value
