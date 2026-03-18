# 1. Import required modules
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

from Ahri.Asuka.config.config import settings

# 2. Initialize the model and load weights
model = Patchcore(pre_trained=False)
engine = Engine()

# 3. Prepare test data
# You can use a single image or a folder of images
dataset = PredictDataset(path=settings.DATA_DIR / "weld/test/defect/defect_0050.png", image_size=(512, 512))

# 4. Get predictions
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=settings.MODELS_DIR / "anomalib/Patchcore/weld/latest/weights/lightning/model.ckpt",
)

# 5. Access the results
if predictions is not None:
    for prediction in predictions:
        image_path = prediction.image_path
        anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
        pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
        pred_score = prediction.pred_score  # Image-level anomaly score
