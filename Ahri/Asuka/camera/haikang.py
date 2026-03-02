"""
使用海康威视摄像头 SDK 读取视频流
下载地址：https://open.hikvision.com/download/5cda567cf47ae80dd41a54b3?type=10

海康没有直接提供 Python SDK 需要自己封装下
"""

import threading
import time
from ctypes import POINTER, byref, c_char, c_long, c_ubyte, cast, create_string_buffer
from types import NoneType

import cv2
import numpy as np
from cv2.typing import MatLike
from HCNetSDK.HCNetSDK import (
    C_LONG,
    NET_DVR_DEVICEINFO_V40,
    NET_DVR_LOCAL_SDK_PATH,
    NET_DVR_PREVIEWINFO,
    NET_DVR_STREAMDATA,
    NET_DVR_SYSHEAD,
    NET_DVR_USER_LOGIN_INFO,
    NET_SDK_INIT_CFG_TYPE,
    REALDATACALLBACK,
    fun_ctype,
    hcnetsdk_path,
    load_library,
    netsdkdllpath,
    playM4dllpath,
    sys_platform,
)
from loguru import logger

from Ahri.Asuka import ASUKA_ROOT

# void (CALLBACK* DisplayCBFun)(long nPort,char * pBuf,long nSize,long nWidth,long nHeight,long nStamp,long nType,long nReserved)
DisplayCBFun = fun_ctype(None, c_long, POINTER(c_char), c_long, c_long, c_long, c_long, c_long, c_long)


class HaikangCamera(threading.Thread):

    def __init__(self, ip: str, port: int, username: str, password: str):
        super().__init__()
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password

        self.log_dir = ASUKA_ROOT / "logs"

        # SDK
        self.hikSDK, self.playM4SDK = self.LoadSDK()  # 加载sdk库
        self.iUserID = -1  # 登录句柄
        self.lRealPlayHandle = -1  # 预览句柄
        # self.wincv = None  # windows环境下的参数
        # self.win = None  # 预览窗口
        self.FuncDecCB = None  # 解码回调
        self.FunDisPlayCB = None
        self.PlayCtrlPort = C_LONG(-1)  # 播放通道号
        # self.basePath = ""  # 基础路径
        # self.preview_file = ""  # linux预览取流保存路径
        self.funcRealDataCallBack_V30 = REALDATACALLBACK(self.RealDataCallBack_V30)  # 预览回调函数
        # self.msg_callback_func = MSGCallBack_V31(self.g_fMessageCallBack_Alarm)  # 注册回调函数实现

        # frame variable
        self.frame: MatLike | NoneType = None

        # thread variable
        self.name = f"{self.__class__.__name__}Thread"
        self.frame_lock = threading.Lock()
        self.open_stream_cond = threading.Condition()
        self.running = True

    def run(self):
        while self.running:
            with self.open_stream_cond:
                if self.open_stream():
                    self.open_stream_cond.wait()

    def open_stream(self):
        self.SetSDKInitCfg()  # 设置SDK初始化依赖库路径
        self.hikSDK.NET_DVR_Init()  # 初始化sdk
        self.GeneralSetting()  # 通用设置，日志，回调函数等
        if not self.LoginDev():  # 登录设备
            return False
        if not self.startPlay(playTime=5):  # playTime用于linux环境控制预览时长，windows环境无效
            return False
        return True

    def release(self):
        with self.open_stream_cond:
            self.running = False
            self.open_stream_cond.notify()
        self.stopPlay()
        self.LogoutDev()
        # 释放资源
        self.hikSDK.NET_DVR_Cleanup()

    def LoadSDK(self):
        hikSDK = None
        playM4SDK = None
        try:
            logger.info(f"{netsdkdllpath=}")
            logger.info(f"{playM4dllpath=}")
            hikSDK = load_library(netsdkdllpath)
            playM4SDK = load_library(playM4dllpath)
        except OSError as e:
            logger.error("动态库加载失败", e)
        return hikSDK, playM4SDK

    def SetSDKInitCfg(self):
        """
        设置 SDK 初始化依赖库路径
        """
        # 设置HCNetSDKCom组件库和SSL库加载路径
        strPath = (
            hcnetsdk_path.encode("gbk") + rb"\lib"
            if sys_platform == "windows"
            else hcnetsdk_path.encode("gbk") + rb"/lib"
        )
        libcrypto_path_buffer = create_string_buffer(
            strPath + rb"\libcrypto-1_1-x64.dll" if sys_platform == "windows" else strPath + rb"/libcrypto.so.1.1"
        )
        libssl_path_buffer = create_string_buffer(
            strPath + rb"\libssl-1_1-x64.dll" if sys_platform == "windows" else strPath + rb"/libssl.so.1.1"
        )

        sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
        sdk_ComPath.sPath = strPath
        logger.info(f"{strPath=}")
        if self.hikSDK.NET_DVR_SetSDKInitCfg(NET_SDK_INIT_CFG_TYPE.NET_SDK_INIT_CFG_SDK_PATH.value, byref(sdk_ComPath)):
            logger.info("HCNetSDKCom 加载成功")
        if self.hikSDK.NET_DVR_SetSDKInitCfg(
            NET_SDK_INIT_CFG_TYPE.NET_SDK_INIT_CFG_LIBEAY_PATH.value, libcrypto_path_buffer
        ):
            logger.info("libcrypto 加载成功")
        if self.hikSDK.NET_DVR_SetSDKInitCfg(
            NET_SDK_INIT_CFG_TYPE.NET_SDK_INIT_CFG_SSLEAY_PATH.value, libssl_path_buffer
        ):
            logger.info("libssl 加载成功")

    def GeneralSetting(self):
        """
        通用设置，日志/回调事件类型等
        日志的等级（默认为0）：
        0-表示关闭日志，
        1-表示只输出ERROR错误日志，
        2-输出ERROR错误信息和DEBUG调试信息，
        3-输出ERROR错误信息、DEBUG调试信息和INFO普通信息等所有信息
        """
        self.hikSDK.NET_DVR_SetLogToFile(3, bytes(str(self.log_dir), encoding="utf-8"), False)

    def LoginDev(self):
        # 登录设备
        # 登录参数，包括设备地址、登录用户、密码等
        struLoginInfo = NET_DVR_USER_LOGIN_INFO()
        struLoginInfo.bUseAsynLogin = 0  # 同步登录方式
        struLoginInfo.sDeviceAddress = self.ip.encode()  # 设备IP地址
        struLoginInfo.wPort = self.port  # 设备服务端口
        struLoginInfo.sUserName = self.username.encode()  # 设备登录用户名
        struLoginInfo.sPassword = self.password.encode()  # 设备登录密码
        struLoginInfo.byLoginMode = 0

        # 设备信息, 输出参数
        struDeviceInfoV40 = NET_DVR_DEVICEINFO_V40()

        self.iUserID = self.hikSDK.NET_DVR_Login_V40(byref(struLoginInfo), byref(struDeviceInfoV40))
        if self.iUserID < 0:
            logger.error("Login failed, error code: %d" % self.hikSDK.NET_DVR_GetLastError())
            self.hikSDK.NET_DVR_Cleanup()
            return False
        else:
            logger.info(f"登录成功，设备序列号：{str(struDeviceInfoV40.struDeviceV30.sSerialNumber, encoding='utf8')}")
            return True

    def LogoutDev(self):
        # 登出设备
        if self.iUserID > -1:
            # 撤销布防，退出程序时调用
            self.hikSDK.NET_DVR_Logout(self.iUserID)

    def DecCBFun(self, nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
        # 解码回调函数
        data = cast(pBuf, POINTER(c_ubyte * nSize)).contents
        info = pFrameInfo.contents

        if info.nType == 3:
            yuv = np.frombuffer(data, dtype=np.uint8).reshape(int(info.nHeight * 1.5), info.nWidth)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YV12)
            self.frame = bgr

    def DisplayCBFun(self, nPort, pBuf, nSize, nWidth, nHeight, nStamp, nType, nReserved):
        """
        void (CALLBACK* DisplayCBFun)(long nPort,char * pBuf,long nSize,long nWidth,long nHeight,long nStamp,long nType,long nReserved)
        """
        data = cast(pBuf, POINTER(c_ubyte * nSize)).contents
        yuv = np.frombuffer(data, dtype=np.uint8).reshape(int(nHeight * 1.5), nWidth)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YV12)
        self.frame = bgr

    def RealDataCallBack_V30(self, lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
        # 码流回调函数
        # if sys_platform == "linux":
        #     # 码流回调函数
        #     if dwDataType == NET_DVR_SYSHEAD:
        #         from datetime import datetime

        #         # 获取当前时间的datetime对象
        #         current_time = datetime.now()
        #         timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
        #         self.preview_file = f"./previewVideo{timestamp_str}.mp4"
        #     # elif dwDataType == NET_DVR_STREAMDATA:
        #     #     self.writeFile(self.preview_file, pBuffer, dwBufSize)
        #     else:
        #         logger.info("其他数据,长度:", dwBufSize)
        # elif sys_platform == "windows":
        if dwDataType == NET_DVR_SYSHEAD:
            # 设置流播放模式
            self.playM4SDK.PlayM4_SetStreamOpenMode(self.PlayCtrlPort, 0)
            # 打开码流，送入40字节系统头数据
            if self.playM4SDK.PlayM4_OpenStream(self.PlayCtrlPort, pBuffer, dwBufSize, 1024 * 1024):
                # 设置解码回调，可以返回解码后YUV视频数据
                # BUG: Windows 可以解码，但是 Linux 上 FRAME_INFO 内的数据都是错误
                # self.FuncDecCB = DECCBFUNWIN(self.DecCBFun)
                # self.playM4SDK.PlayM4_SetDecCallBackExMend(self.PlayCtrlPort, self.FuncDecCB, None, 0, None)
                # NOTE: Windows/Linux 都可以解码
                self.FunDisPlayCB = DisplayCBFun(self.DisplayCBFun)
                self.playM4SDK.PlayM4_SetDisplayCallBack(self.PlayCtrlPort, self.FunDisPlayCB)
                # 开始解码播放
                if self.playM4SDK.PlayM4_Play(self.PlayCtrlPort, 0):
                    logger.info("播放库播放成功")
                else:
                    logger.info("播放库播放失败")
            else:
                logger.info(f"播放库打开流失败, 错误码：{self.playM4SDK.PlayM4_GetLastError(self.PlayCtrlPort)}")
        elif dwDataType == NET_DVR_STREAMDATA:
            self.playM4SDK.PlayM4_InputData(self.PlayCtrlPort, pBuffer, dwBufSize)
        else:
            logger.info("其他数据,长度:", dwBufSize)

    def startPlay(self, playTime):
        # 获取一个播放句柄
        if not self.playM4SDK.PlayM4_GetPort(byref(self.PlayCtrlPort)):
            logger.info(f"获取播放库句柄失败, 错误码：{self.playM4SDK.PlayM4_GetLastError(self.PlayCtrlPort)}")
            return False

        # 开始预览
        preview_info = NET_DVR_PREVIEWINFO()
        preview_info.hPlayWnd = 0
        preview_info.lChannel = 1  # 通道号
        preview_info.dwStreamType = 0  # 主码流
        preview_info.dwLinkMode = 0  # TCP
        preview_info.bBlocked = 1  # 阻塞取流

        # 开始预览并且设置回调函数回调获取实时流数据
        self.lRealPlayHandle = self.hikSDK.NET_DVR_RealPlay_V40(
            self.iUserID, byref(preview_info), self.funcRealDataCallBack_V30, None
        )
        if self.lRealPlayHandle < 0:
            logger.info("Open preview fail, error code is: %d" % self.hikSDK.NET_DVR_GetLastError())
            # 登出设备
            self.hikSDK.NET_DVR_Logout(self.iUserID)
            # 释放资源
            self.hikSDK.NET_DVR_Cleanup()
            return False

        if sys_platform == "linux":
            time.sleep(playTime)

        return True

    def stopPlay(self):
        # 关闭预览
        self.hikSDK.NET_DVR_StopRealPlay(self.lRealPlayHandle)

        # 停止解码，释放播放库资源
        if self.PlayCtrlPort.value > -1:
            self.playM4SDK.PlayM4_Stop(self.PlayCtrlPort)
            self.playM4SDK.PlayM4_CloseStream(self.PlayCtrlPort)
            self.playM4SDK.PlayM4_FreePort(self.PlayCtrlPort)
            self.PlayCtrlPort = C_LONG(-1)


def main():
    camera = HaikangCamera("192.168.30.111", 8000, "admin", "linxin789")
    # camera = HaikangCamera("192.168.8.68", 8000, "admin", "linxin789")
    camera.start()

    cv2.namedWindow("Haikang Camera", cv2.WINDOW_FREERATIO)

    while True:
        if camera.frame is not None:
            copy_frame = camera.frame.copy()
            # time_text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # cv2.putText(copy_frame, f"{time_text}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.imshow("Haikang Camera", copy_frame)
        if cv2.waitKey(1) == 27:  # ESC 键退出
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
