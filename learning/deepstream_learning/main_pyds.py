# ruff: noqa: E402

"""
DeepStream(pyds) using GStreamer API
"""

import gi

gi.require_version('Gst', '1.0')  # 使用 GStreamer API 1.0

import contextlib
import math
import queue
import signal
import sys
import threading
import time
from typing import Any

import gi
import numpy as np
import pyds
from gi.repository import GLib, Gst  # pyright: ignore[reportMissingModuleSource]
from loguru import logger

from common import PERF_DATA, PlatformInfo, bus_call, long_to_uint64

PGIE_CONFIG_FILE = "/workspace/DeepStream-Yolo/config_infer_primary_yolo26.txt"
# nvtracker, 也适用于 YOLO
TRACKER_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
SUPPORT_URI_PREFIXES = ('rtsp://', 'http://', 'https://', 'file://')

STREAMS = [
    (1, "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"),
    (2, "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"),
    (3, "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"),
    # add more streams here
]


class DeepStreamPipeline(threading.Thread):

    def __init__(
        self,
        streams: list[tuple[str, str]],
        detection_queue: queue.Queue,
        width: int = 640,
        height: int = 480,
        live_source: bool = True,
        display: bool = False,
        display_width: int = 1920,
        display_height: int = 1080,
    ):
        super().__init__(daemon=True)
        self.streams = streams
        self.cam_ids = [str(cam_id) for cam_id, _ in streams]
        self.uris = [url for _, url in streams]
        self.detection_queue = detection_queue
        self.width = width
        self.height = height
        self.live_source = live_source
        self.display = display
        self.display_width = display_width
        self.display_height = display_height

        self.num_sources = len(self.uris)
        self.pipeline = None
        self.loop = None

        self.perf_data = PERF_DATA(streams)

    def stop(self):
        if self.loop:
            GLib.idle_add(self.loop.quit)
        self.join(timeout=5)
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

    def run(self):
        Gst.init(None)
        self.pipeline = self._build_pipeline()
        if not self.pipeline:
            logger.error("Failed to build pipeline")
            return

        # 总线消息处理
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, self.loop)

        GLib.timeout_add_seconds(5, self.perf_data.perf_print_callback)

        # 启动管道
        self.pipeline.set_state(Gst.State.PLAYING)
        logger.info("DeepStream pipeline started")
        try:
            self.loop.run()
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def _build_pipeline(self):  # noqa: C901
        pipeline = Gst.Pipeline(name="deepstream-pipeline")
        if not pipeline:
            logger.error("Unable to create pipeline")
            return None

        # 1. 创建 nvstreammux
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            logger.error("Failed to create nvstreammux")
            return None
        pipeline.add(streammux)
        streammux.set_property("width", self.width)
        streammux.set_property("height", self.height)
        streammux.set_property("batch-size", self.num_sources)
        streammux.set_property("batched-push-timeout", 33000)
        if self.live_source:
            streammux.set_property("live-source", 1)
        # 关闭强制同步，避免等待所有源对齐时间戳
        streammux.set_property("sync-inputs", 0)

        # 2. 为每个 Stream 创建 source bin 并连接到 streammux
        for i, url in enumerate(self.uris):
            source_bin = self._create_source_bin(i, url)
            if not source_bin:
                logger.error(f"Failed to create source bin for {url}")
                continue
            pipeline.add(source_bin)
            sinkpad = streammux.request_pad_simple(f"sink_{i}")
            srcpad = source_bin.get_static_pad("src")
            if not sinkpad or not srcpad:
                logger.error(f"Unable to get pads for source {i}")
                continue
            srcpad.link(sinkpad)

        # 3. 推理引擎 nvinfer
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            logger.error("Failed to create pgie")
            return None
        pgie.set_property("config-file-path", PGIE_CONFIG_FILE)
        pgie.set_property("batch-size", self.num_sources)
        pipeline.add(pgie)

        # 4. 跟踪器 nvtracker
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            logger.error("Failed to create tracker")
            return None
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", 0)
        tracker.set_property("ll-lib-file", TRACKER_LIB_FILE)
        tracker.set_property("ll-config-file", TRACKER_CONFIG_FILE)
        pipeline.add(tracker)

        streammux.link(pgie)
        pgie.link(tracker)

        # 5. 显示 Display
        if self.display:
            tiler_rows = int(math.sqrt(self.num_sources))
            tiler_columns = math.ceil(self.num_sources / tiler_rows)
            tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
            if not tiler:
                logger.error("Unable to create tiler")
                return None
            tiler.set_property("rows", tiler_rows)
            tiler.set_property("columns", tiler_columns)
            tiler.set_property("width", self.display_width)
            tiler.set_property("height", self.display_height)

            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

            sink = self._get_osd_sink()
            if not sink:
                logger.error("Unable to create sink element")
                return None

            pipeline.add(tiler)
            pipeline.add(nvvidconv)
            pipeline.add(nvosd)
            pipeline.add(sink)

            tracker.link(nvvidconv)
            nvvidconv.link(tiler)
            tiler.link(nvosd)
            nvosd.link(sink)
        else:
            # 无显示：使用 fakesink
            sink = Gst.ElementFactory.make("fakesink", "fakesink")
            sink.set_property("sync", 0)
            pipeline.add(sink)
            tracker.link(sink)

        # 6. 探针 probe
        pgie_src_pad = pgie.get_static_pad("src")
        if pgie_src_pad:
            pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._probe_callback, 0)
        else:
            logger.error("Warning: Could not get pgie src pad")
            return None
        return pipeline

    def _get_osd_sink(self):
        platform_info = PlatformInfo()
        if platform_info.is_integrated_gpu():
            logger.info("Creating nv3dsink")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                logger.error("Unable to create nv3dsink")
        else:
            if platform_info.is_platform_aarch64():
                logger.info("Creating nv3dsink")
                sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            else:
                logger.info("Creating EGLSink")
                sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        return sink

    def _create_source_bin(self, index: int, uri: str):  # noqa: C901
        if not uri.startswith(SUPPORT_URI_PREFIXES):
            uri = 'file://' + uri
        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            logger.error("Unable to create source bin")
            return None

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
        if not uri_decode_bin:
            logger.error("Unable to create uri decode bin")
            return None
        # We set the input uri to the source element
        uri_decode_bin.set_property("uri", uri)

        def cb_newpad(decodebin: Gst.Element, decoder_src_pad: Gst.Pad, data: Gst.Bin):
            caps = decoder_src_pad.get_current_caps()
            gststruct = caps.get_structure(0)
            gstname = gststruct.get_name()
            source_bin = data
            features = caps.get_features(0)

            # Need to check if the pad created by the decodebin is for video and not
            # audio.
            if gstname.find("video") != -1:
                # Link the decodebin pad only if decodebin has picked nvidia
                # decoder plugin nvdec_*. We do this by checking if the pad caps contain
                # NVMM memory features.
                if features.contains("memory:NVMM"):
                    # Get the source bin ghost pad
                    bin_ghost_pad = source_bin.get_static_pad("src")
                    if not bin_ghost_pad.set_target(decoder_src_pad):
                        logger.error("Failed to link decoder src pad to source bin ghost pad")
                else:
                    logger.error("Error: Decodebin did not pick nvidia decoder plugin")

        def decodebin_child_added(child_proxy: Gst.Object, Object: Gst.Element, name: str, user_data: Gst.Bin):
            if name.find("decodebin") != -1:
                Object.connect("child-added", decodebin_child_added, user_data)

            # if ts_from_rtsp:
            #     if name.find("source") != -1:
            #         pyds.configure_source_for_ntp_sync(hash(Object))

        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            logger.error("Failed to add ghost pad in source bin")
            return None
        return nbin

    def _probe_callback(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: Any):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            source_id = frame_meta.pad_index
            if source_id >= len(self.cam_ids):
                l_frame = l_frame.next
                continue
            cam_id = self.cam_ids[source_id]

            # 性能统计
            self.perf_data.update_fps(cam_id)

            # 时间戳
            timestamp = frame_meta.ntp_timestamp / 1000000000.0 if frame_meta.ntp_timestamp != 0 else time.time()

            boxes_list = []
            class_ids_list = []
            track_ids_list = []
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                left = obj_meta.rect_params.left
                top = obj_meta.rect_params.top
                w = obj_meta.rect_params.width
                h = obj_meta.rect_params.height
                boxes_list.append([left, top, left + w, top + h])
                class_ids_list.append(obj_meta.class_id)
                track_ids_list.append(long_to_uint64(obj_meta.object_id))
                l_obj = l_obj.next

            if boxes_list:
                boxes = np.array(boxes_list, dtype=np.float32)
                class_ids = np.array(class_ids_list, dtype=np.int32)
                track_ids = np.array(track_ids_list, dtype=np.uint64)
                with contextlib.suppress(queue.Full):
                    self.detection_queue.put_nowait(
                        {
                            "cam_id": cam_id,
                            "boxes": boxes,
                            "class_ids": class_ids,
                            "track_ids": track_ids,
                            "timestamp": timestamp,
                        }
                    )

            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK


def main():
    detection_queue = queue.Queue(maxsize=100)
    pipeline = DeepStreamPipeline(
        streams=STREAMS,
        detection_queue=detection_queue,
        width=640,
        height=480,
        live_source=True,
        display=True,
        display_width=1280,
        display_height=720,
    )

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    pipeline.start()

    def consume_detections():
        while True:
            try:
                det = detection_queue.get(timeout=1)
                logger.debug(f"Received detection: {det['cam_id']} -> {len(det['boxes'])} objects")
            except queue.Empty:
                continue

    consumer_thread = threading.Thread(target=consume_detections, daemon=True)
    consumer_thread.start()

    pipeline.join()


if __name__ == "__main__":
    main()
