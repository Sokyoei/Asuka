"""
ERROR:

The model and loaded state dict do not match exactly

unexpected key in source state_dict: data_preprocessor.mean, data_preprocessor.std
"""

import subprocess
from pathlib import Path

import mmdet
from mmdet.apis import inference_detector, init_detector

from Ahri.Asuka import SOKYOEI_DATA_DIR

MMDET_ROOT = Path(mmdet.__file__).parent


def main():
    subprocess.run(["mim", "download", "mmdet", "--config", "rtmdet_tiny_8xb32-300e_coco", "--dest", "."])
    config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
    inference_detector(model, str(SOKYOEI_DATA_DIR / 'Ahri/Popstar Ahri.jpg'))


if __name__ == "__main__":
    main()
