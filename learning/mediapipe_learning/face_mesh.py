import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# eye index
# see: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


def inference_images():
    IMAGE_FILES = []
    _drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        for _idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)


def inference_camera(index=0):
    _drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(index)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            height, width, _ = image.shape
            eye_points = np.array([])
            points = np.array([])

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    eye_points = np.array([face_landmarks.landmark[i] for i in LEFT_EYE_INDICES + RIGHT_EYE_INDICES])
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    # )
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    # )
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    # )
                points = np.array(
                    [
                        [landmark.x * width, landmark.y * height, landmark.visibility]
                        for _, landmark in enumerate(eye_points)
                    ]
                )
                for p in points:
                    cv2.circle(image, [int(p[0]), int(p[1])], 2, (255, 0, 0), 2)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def main():
    # inference_images()
    inference_camera(0)


if __name__ == "__main__":
    main()
