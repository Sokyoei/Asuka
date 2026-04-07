import os
from pathlib import Path

ASUKA_ROOT = Path(__file__).resolve().parent.parent.parent

sokyoei_data_dir = os.environ.get('SOKYOEI_DATA_DIR')
if sokyoei_data_dir:
    SOKYOEI_DATA_DIR = Path(sokyoei_data_dir)
else:
    raise RuntimeError("Please install https://github.com/Sokyoei/data")

from Ahri.Asuka.logger import init_logging  # noqa: E402

init_logging()
