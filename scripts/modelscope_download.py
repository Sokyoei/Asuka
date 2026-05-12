"""
魔搭社区(https://modelscope.cn/home)下载大模型
"""

from modelscope import snapshot_download

from Ahri.Asuka.config.config import settings

snapshot_download("your_model_repo_name", cache_dir=settings.MODELS_DIR)
