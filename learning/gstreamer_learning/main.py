# ruff: noqa: E402

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# 初始化 GStreamer
Gst.init(None)
print(f"Hello, GStreamer! Version: {Gst.version_string()}")

Gst.deinit()
