# 1. Import required modules
import os

import safetensors.torch
from anomalib.data import Folder, MVTecAD  # noqa: F401
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

from Ahri.Asuka.config.config import settings

# huggingface China mirror
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

ANOMALIB_MODELS_DIR = settings.MODELS_DIR / "anomalib"
# 2. Create a dataset
# MVTecAD is a popular dataset for anomaly detection
# datamodule = MVTecAD(
#     root=settings.DATA_DIR / "MVTecAD",  # Path to download/store the dataset
#     category="bottle",  # MVTec category to use
#     train_batch_size=32,  # Number of images per training batch
#     eval_batch_size=32,  # Number of images per validation/test batch
# )
dataset_dir = settings.DATA_DIR / "weld"
datamodule = Folder(name="weld", root=dataset_dir, normal_dir=dataset_dir / "train", abnormal_dir=dataset_dir / "test")

# 3. Initialize the model
# Patchcore is a good choice for beginners
model = Patchcore(num_neighbors=9, pre_trained=False)  # Override default model settings
# https://huggingface.co/timm/wide_resnet50_2.racm_in1k/resolve/main/model.safetensors
state_dict = safetensors.torch.load_file(settings.MODELS_DIR / "wide_resnet50_2.racm_in1k.safetensors")
model.model.feature_extractor.load_state_dict(state_dict, strict=False)

# 4. Create the training engine
engine = Engine(
    max_epochs=1,  # train epochs
    accelerator="auto",  # auto detect device CPU/GPU
    devices=1,
    default_root_dir=ANOMALIB_MODELS_DIR,
)  # Override default trainer settings

# 5. Train the model
# This produces a lightning model (.ckpt)
engine.fit(datamodule=datamodule, model=model)

# 6. Test the model performance
test_results = engine.test(datamodule=datamodule, model=model)

# 7. Export the model
# Different formats are available: Torch, OpenVINO, ONNX
engine.export(model=model, export_root=ANOMALIB_MODELS_DIR, export_type=ExportType.OPENVINO)
