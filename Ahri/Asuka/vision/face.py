import os
from enum import IntEnum
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from Ahri.Asuka import ASUKA_ROOT
from Ahri.Asuka.config.config import settings
from Ahri.Asuka.constants import IMAGE_SUFFIXES
from Ahri.Asuka.utils.cv2_utils import imread

TEMPLATES_DIR = settings.IMAGES_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def _cal_distance(embedding1, embedding2, threshold=0.65):
    """计算余弦相似度"""
    cosine_distance = 1 - np.dot(embedding1, b=embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    similarity = 1 - (cosine_distance / 2)
    return similarity >= threshold, similarity


class FaceType(IntEnum):
    deepface = 1  # https://github.com/serengil/deepface
    insightface = 2  # https://github.com/deepinsight/insightface


class DeepFaceRecognition(object):
    from deepface import DeepFace

    def __init__(
        self, model_name: str = "Facenet512", detector_backend: str = "retinaface", preson_face_confidence: float = 0.9
    ):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.preson_face_confidence = preson_face_confidence
        self.features = {}  # {人名: 特征向量}

    def add_template_faces(self, template_dir: os.PathLike | str = TEMPLATES_DIR):
        """添加模板人脸"""
        for file in Path(template_dir).iterdir():
            if file.suffix in IMAGE_SUFFIXES:
                face_name = file.stem
                face_image = imread(str(file))

                if face_image is None:
                    logger.error(f"无法读取图片: {file}")
                    continue

                # 提取人脸特征
                face_embedding = self.DeepFace.represent(face_image, self.model_name, False, self.detector_backend)
                # 人脸置信度排序
                sorted_face_embedding = sorted(face_embedding, key=lambda x: x["face_confidence"], reverse=True)
                if sorted_face_embedding[0]["face_confidence"] > self.preson_face_confidence:
                    face_embedding = sorted_face_embedding[0]["embedding"]
                    self.features[face_name] = face_embedding
                    logger.info(f"已添加模板人脸: {face_name}")
                else:
                    logger.warning(f"{file} 未识别到人脸！")

        if not self.features:
            logger.warning("未添加任何模板人脸!")
            return False

        return True

    def compare_faces(self, frame: NDArray, name: str, threshold: float = 0.65):
        """人脸识别"""
        results = []
        # 获取模板特征
        template_embedding = self.features.get(name)
        if template_embedding is None:
            return [[0.0, [0, 0, 0, 0]]]

        # 提取带检测图像的人脸特征
        target_embedding = self.DeepFace.represent(frame, self.model_name, False, self.detector_backend)
        target_embedding = [info for info in target_embedding if info["face_confidence"] > self.preson_face_confidence]
        for info in target_embedding:
            target_embedding = info["embedding"]
            is_match, score = _cal_distance(template_embedding, target_embedding, threshold)
            if is_match:
                x, y, w, h = (
                    info["facial_area"]["x"],
                    info["facial_area"]["y"],
                    info["facial_area"]["w"],
                    info["facial_area"]["h"],
                )
                bbox = [x, y, x + w, y + h]
                results.append([score, bbox])

        return results if results else [[0.0, [0, 0, 0, 0]]]


class InsightFaceRecognition(object):
    from insightface.app import FaceAnalysis

    def __init__(
        self,
        name='buffalo_l',
        root=ASUKA_ROOT,
        det_thresh=0.5,
        det_size=(640, 640),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        ctx_id=0,  # GPU id, -1 CPU
    ):
        self.app = self.FaceAnalysis(name=name, providers=providers, root=root)
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
        self.features = {}  # {人名: 特征向量}

    def add_template_faces(self, template_dir: os.PathLike | str = TEMPLATES_DIR):
        for file in Path(template_dir).iterdir():
            if file.suffix.lower() in IMAGE_SUFFIXES:
                face_name = file.stem  # 用文件名作为人脸标识
                face_image = imread(str(file))

                if face_image is None:
                    logger.error(f"无法读取图片: {file}")
                    continue

                # 检测人脸
                faces = self.app.get(face_image)
                if not faces:
                    logger.warning(f"未检测到人脸: {file}")
                    continue

                embedding = faces[0]['embedding'].astype('float32')
                self.features[face_name] = embedding
                logger.info(f"已添加模板人脸: {face_name}")

        if not self.features:
            logger.warning("未添加任何模板人脸!")
            return False

        return True

    def compare_faces(self, frame: NDArray, name: str, threshold: float = 0.65):
        results = []

        template_embedding = self.features.get(name)
        if template_embedding is None:
            return [[0.0, [0, 0, 0, 0]]]

        faces = self.app.get(frame)
        if faces is None:
            return [[0.0, [0, 0, 0, 0]]]

        for face in faces:
            target_embedding = face['embedding'].astype('float32')
            is_match, score = _cal_distance(template_embedding, target_embedding, threshold)
            if is_match:
                x1, y1, x2, y2 = map(int, face['bbox'])
                results.append([score, [x1, y1, x2, y2]])

        return results if results else [[0.0, [0, 0, 0, 0]]]
