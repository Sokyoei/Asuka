"""
使用大华摄像头 SDK 进行视频流采集
SDK 下载地址：https://support.dahuatech.com/sdkindex/sdkExploit

## ARM 版 NetSDK

ARM 版的需要下载设备网络SDK和播放SDK, 将 linux x64 平台的 python 包里的 so 替换为这两个包的 so,
然后按照下面注释掉 NetSDK.py/NetClient/_load_library() 函数内的两行，再重新打包一下

```python
    @classmethod
    def _load_library(cls):
        try:
            cls.sdk = load_library(netsdkdllpath)
            cls.config_sdk = load_library(configdllpath)
            cls.render_sdk = load_library(rendersdkdllpath)
            # cls.infra_sdk = load_library(infrasdkdllpath)  # 注释掉
            # cls.image_alg = load_library(imagealgdllpath)  # 注释掉
            cls.play_sdk = load_library(playsdkdllpath)
        except OSError as e:
            print('动态库加载失败')
```
"""

import datetime
import os
import platform
import site
import threading
from ctypes import CDLL, POINTER, c_char_p, c_int, c_long, c_ubyte, cast, sizeof

# Linux 需要将 SDK 的动态库路径加入到 LD_LIBRARY_PATH 环境变量，Python 的 os.environ 不起作用
#
# ```shell
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/miniconda3/envs/health/lib/python3.10/site-packages/NetSDK/Libs/linux64
# ```
#
# VVV 无效 VVV
if platform.system() == "Linux":
    os.environ["LD_LIBRARY_PATH"] += f":{site.getsitepackages()[0]}/NetSDK/Libs/linux64"
    print(os.environ["LD_LIBRARY_PATH"])
    CDLL(f"{site.getsitepackages()[0]}/NetSDK/Libs/linux64/libRenderEngine.so")
    CDLL(f"{site.getsitepackages()[0]}/NetSDK/Libs/linux64/libplay.so")

import cv2
import numpy as np
from loguru import logger
from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDecCallBack, fDecCBFun, fDisConnect, fHaveReConnect, fRealDataCallBackEx2
from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE, EM_REALDATA_FLAG, NET_VIDEOSTREAM_TYPE, SDK_RealPlayType
from NetSDK.SDK_Struct import (
    C_LLONG,
    LOG_SET_PRINT_INFO,
    NET_FRAME_DECODE_INFO,
    NET_FRAME_INFO_EX,
    NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
    NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
    NET_TIME_EX,
)

from Ahri.Asuka import ASUKA_ROOT


def decode_frame_to_bgr(frame_info):
    # 获取 Y、U、V 三个分量指针
    y_ptr, u_ptr, v_ptr = frame_info.pVideoData
    y_stride, u_stride, v_stride = frame_info.nStride
    y_width, u_width, v_width = frame_info.nWidth
    y_height, u_height, v_height = frame_info.nHeight

    # 获取图像尺寸（以 Y 为准）
    height = y_height
    width = y_width

    # 读取每个分量数据为 numpy 数组（按行读取）
    def extract_plane(ptr, stride, width, height):
        buf = cast(ptr, POINTER(c_ubyte * (stride * height))).contents
        raw = np.frombuffer(buf, dtype=np.uint8).reshape(height, stride)
        return raw[:, :width]  # 只取有效区域

    y = extract_plane(y_ptr, y_stride, y_width, y_height)
    u = extract_plane(u_ptr, u_stride, u_width, u_height)
    v = extract_plane(v_ptr, v_stride, v_width, v_height)

    # 将 YUV420p 重新打包为连续内存（Y + U + V）
    yuv420p = np.concatenate([y.flatten(), u.flatten(), v.flatten()]).astype(np.uint8)

    # 转换为 OpenCV 图像
    yuv_img = yuv420p.reshape((int(height * 1.5), width))
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_I420)

    return bgr_img


class DahuaCamera(threading.Thread):

    def __init__(self, ip: str, port: int, username: str, password: str):
        super().__init__()
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password

        # config variable
        self.log_file = ASUKA_ROOT / "logs/dahua_camera.log"

        # SDK return value
        self.login_id = C_LLONG()
        self.play_id = C_LLONG()
        self.freeport = c_int()
        # SDK constant value
        self.streamtype = SDK_RealPlayType.Realplay
        # SDK callback function
        self.m_DisConnectCallBack = fDisConnect(self.DisConnectCallBack)
        self.m_ReConnectCallBack = fHaveReConnect(self.ReConnectCallBack)
        self.m_RealDataCallBack = fRealDataCallBackEx2(self.RealDataCallBack)
        # fDecCBFun = CB_FUNCTYPE(None, c_int, c_void_p, c_int, POINTER(PLAY_FRAME_INFO), c_void_p, c_int)
        self.m_DecodingCallBack = fDecCBFun(self.DecodingCallBack)
        # fDecCallBack = CB_FUNCTYPE(None, C_LLONG, C_LLONG, POINTER(NET_FRAME_DECODE_INFO), POINTER(NET_FRAME_INFO_EX), C_LDWORD, C_LLONG)
        self.m_DecodingCallBackEx = fDecCallBack(self.DecodingCallBackEx)
        # SDK client
        self.sdk = NetClient()
        self.sdk.InitEx(self.m_DisConnectCallBack)
        self.sdk.SetAutoReconnect(self.m_ReConnectCallBack)

        self.init_log()
        self.init_login()

        # frame variable
        self.frame = None

        # thread variable
        self.name = f"{self.__class__.__name__}Thread"
        self.frame_lock = threading.Lock()
        self.open_stream_cond = threading.Condition()
        self.running = True

    def open_stream(self) -> bool:
        """打开视频流

        Returns:
            bool: True is successed, False is failed.
        """
        result, self.freeport = self.sdk.GetFreePort()
        if result == 0:
            return False

        # self.sdk.SetDecCallBackEx(self.m_DecodingCallBackEx, 0, NET_VIDEOSTREAM_TYPE.VIDEOSTREAM_NORMAL, 0)
        ret = self.sdk.SetDecCallBackEx(self.m_DecodingCallBackEx, 0, NET_VIDEOSTREAM_TYPE.VIDEOSTREAM_NORMAL, 0)
        logger.info(f"SetDecCallBackEx result: {ret}, error message: {self.sdk.GetLastError()}")

        self.sdk.OpenStream(self.freeport)
        self.sdk.Play(self.freeport, 0)
        self.play_id = self.sdk.RealPlayEx(self.login_id, 0, 0, self.streamtype)
        self.sdk.SetRealDataCallBackEx2(
            self.play_id,
            self.m_RealDataCallBack,
            None,
            EM_REALDATA_FLAG.RAW_DATA | EM_REALDATA_FLAG.DATA_WITH_FRAME_INFO,
            # EM_REALDATA_FLAG.RAW_DATA,
        )
        self.sdk.SetDecCallBack(self.freeport, self.m_DecodingCallBack)
        # ret = self.sdk.SetDecCallBackEx(self.m_DecodingCallBackEx, 0, NET_VIDEOSTREAM_TYPE.VIDEOSTREAM_NORMAL, 0)
        # logger.info(f"SetDecCallBackEx result: {ret}")
        logger.info("Dahua camera updated.")
        return True

    def run(self):
        while self.running:
            with self.open_stream_cond:
                if self.open_stream():
                    self.open_stream_cond.wait()

    def release(self):
        with self.open_stream_cond:
            self.running = False
            self.open_stream_cond.notify()
        if self.play_id:
            self.sdk.StopRealPlayEx(self.play_id)
            self.play_id = 0
        if self.freeport:
            self.sdk.SetDecCallBack(self.freeport, None)
            self.sdk.Stop(self.freeport)
            self.sdk.CloseStream(self.freeport)
            self.sdk.ReleasePort(self.freeport)
        if self.login_id:
            self.sdk.Logout(self.login_id)
            self.login_id = 0
        logger.info("Dahua camera released.")

    def DisConnectCallBack(self, lLoginID, pchDVRIP: c_char_p, nDVRPort: c_long, dwUser):
        logger.info("设备断线")

    def ReConnectCallBack(self, lLoginID, pchDVRIP: c_char_p, nDVRPort: c_long, dwUser):
        logger.info("设备重连")

    def RealDataCallBack(self, lRealHandle, dwDataType, pBuffer, dwBufSize, param, dwUser):
        # logger.info(f"RealDataCallBack: {lRealHandle=}, {dwDataType=}, {pBuffer=}, {dwBufSize=}, {param=}, {dwUser=}")
        if lRealHandle != self.play_id or not pBuffer:
            return

        # param_size = sizeof(FRAME_INFO_EX)
        # raw_data = (c_ubyte * param_size).from_address(param)
        # logger.info(f"Raw param data: {bytes(raw_data).hex()}")

        # if dwDataType == (EM_REALDATA_FLAG.DATA_WITH_FRAME_INFO | EM_REALDATA_FLAG.RAW_DATA):
        #     if param:
        #         try:
        #             frame_info_ex = cast(param, POINTER(FRAME_INFO_EX)).contents
        #             ts = frame_info_ex.stTime
        #             timestamp_str = (
        #                 f"{ts.dwYear:04d}-{ts.dwMonth:02d}-{ts.dwDay:02d} "
        #                 f"{ts.dwHour:02d}:{ts.dwMinute:02d}:{ts.dwSecond:02d}.{ts.dwMillisecond:03d}"
        #             )
        #             logger.info(f"[帧时间戳] {timestamp_str}")
        #         except Exception as e:
        #             logger.warning(f"解析帧信息失败: {e}")

        # logger.info(f"InputData: {dwBufSize} bytes")
        self.sdk.InputData(self.freeport, pBuffer, dwBufSize)  # 原始视频流送播放库

    def DecodingCallBackEx(self, lLoginID, lPlayHandle, pFrameDecodeInfo, pFrameInfo, dwUserData, nReserved):
        """
        函数原型

        ```c
        typedef void (CALLBACK *fDecCallBack)(LLONG lLoginID, LLONG lPlayHandle, NET_FRAME_DECODE_INFO* pFrameDecodeInfo, NET_FRAME_INFO_EX* pFrameInfo, LDWORD dwUserData, LLONG nReserved);
        ```
        """
        logger.info(
            f"DecodingCallBackEx: {lLoginID=}, {lPlayHandle=}, {pFrameDecodeInfo=}, {pFrameInfo=}, {dwUserData=}, {nReserved=}"
        )
        decode_info = cast(pFrameDecodeInfo, POINTER(NET_FRAME_DECODE_INFO)).contents
        bgr = decode_frame_to_bgr(decode_info)
        ex_info = cast(pFrameInfo, POINTER(NET_FRAME_INFO_EX)).contents
        time = self.parse_frame_time(ex_info.nDataTime)
        logger.info(f"{time=}")
        self.frame = bgr

    def DecodingCallBack(self, nPort, pBuf, nSize, pFrameInfo, pUserData, nReserved2):
        # logger.info(f"DecodingCallBack: {nPort=} {pBuf=} {nSize=} {pFrameInfo=} {pUserData=} {nReserved2=}")
        # 帧信息结构体
        # here get YUV data, pBuf is YUV data IYUV/YUV420 ,size is nSize, pFrameInfo is frame info with height, width.
        # 对于planar 的YUV格式，先连续存储所有像素点的Y，紧接着存储所有像素点的U，随后是V
        # 对于packed 的YUV格式，每个像素点的Y,U,V是连续交叉存储的
        # uv 的排列格式分为p、sp ，错误会导致颜色不对
        data = cast(pBuf, POINTER(c_ubyte * nSize)).contents
        # print(data)
        info = pFrameInfo.contents
        # info.nType == 3 is YUV data,others ard audio data.
        # you can parse YUV420 data to RGB
        if info.nType == 3:
            # # 使用 numpy.frombuffer 将 c_ubyte_Array 对象转换为 numpy 数组
            yuv = np.frombuffer(data, dtype=np.uint8).reshape(int(info.nHeight * 1.5), info.nWidth)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            # # 如果需要显示图像，可以使用以下代码
            self.frame = bgr

    def parse_frame_time(self, time_struct: NET_TIME_EX) -> str:
        """将NET_TIME_EX转换为可读时间字符串"""
        return (
            f"{time_struct.dwYear:04}-{time_struct.dwMonth:02}-{time_struct.dwDay:02} "
            f"{time_struct.dwHour:02}:{time_struct.dwMinute:02}:"
            f"{time_struct.dwSecond:02}.{time_struct.dwMillisecond:03}"
        )

    def init_log(self):
        # 每次只记录当前的日志
        if self.log_file.exists():
            self.log_file.unlink()
        log_info = LOG_SET_PRINT_INFO()
        log_info.dwSize = sizeof(LOG_SET_PRINT_INFO)
        log_info.bSetFilePath = 1
        log_info.szLogFilePath = str(self.log_file).encode('gbk')
        self.sdk.LogOpen(log_info)

    def init_login(self):
        if not self.login_id:
            stuInParam = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
            stuInParam.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
            stuInParam.szIP = self.ip.encode()
            stuInParam.nPort = self.port
            stuInParam.szUserName = self.username.encode()
            stuInParam.szPassword = self.password.encode()
            stuInParam.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
            stuInParam.pCapParam = None

            stuOutParam = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
            stuOutParam.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)

            self.login_id, device_info, error_msg = self.sdk.LoginWithHighLevelSecurity(stuInParam, stuOutParam)
            if self.login_id != 0:
                logger.info(f"Login succeed. Channel num: {device_info.nChanNum}")
                return True
            else:
                logger.error(f"Login failed, reason: {error_msg}")
                return False
        return False


def main():
    n = 0
    # camera = DahuaCamera("192.168.8.97", 37777, "admin", "L23C0A16")
    camera = DahuaCamera("192.168.1.108", 37777, "admin", "linxin789")
    camera.start()

    cv2.namedWindow("Dahua Camera", cv2.WINDOW_FREERATIO)

    while True:
        if camera.frame is not None:
            copy_frame = camera.frame.copy()
            time_text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(copy_frame, f"{time_text}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            # cv2.imwrite(f"frame/frame_{n}.jpg", copy_frame)
            cv2.imshow("Dahua Camera", copy_frame)
            n += 1
        if cv2.waitKey(1) == 27:  # ESC 键退出
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
