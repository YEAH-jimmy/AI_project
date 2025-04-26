import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# === Mediapipe 초기화 ===
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# 얼굴 검출기
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# 배경제거 모델
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def extract_faces_with_segmentation(input_root, output_root, face_size=(224, 224), margin_ratio=0.4, min_face_size=50):
    os.makedirs(output_root, exist_ok=True)

    for folder_name in sorted(os.listdir(input_root)):
        subdir = os.path.join(input_root, folder_name)
        if not os.path.isdir(subdir):
            continue

        save_subdir = os.path.join(output_root, folder_name)
        os.makedirs(save_subdir, exist_ok=True)

        image_files = [f for f in os.listdir(subdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for idx, file in enumerate(tqdm(image_files, desc=f"Processing {folder_name}")):
            input_path = os.path.join(subdir, file)
            output_path = os.path.join(save_subdir, f"face_{idx:04d}.png")

            frame = cv2.imread(input_path)
            if frame is None:
                print(f"⚠️ 이미지를 불러오지 못함: {input_path}")
                continue

            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)

            if not results.detections:
                print(f"😢 얼굴 검출 실패: {input_path}")
                continue

            # 가장 큰 얼굴만 사용
            biggest_face = None
            max_area = 0
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
                area = w * h
                if area > max_area:
                    max_area = area
                    biggest_face = bboxC

            if biggest_face is None:
                print(f"😢 얼굴 박스 찾기 실패: {input_path}")
                continue

            # 절대 좌표 변환
            x1 = int((biggest_face.xmin - margin_ratio * biggest_face.width) * width)
            y1 = int((biggest_face.ymin - margin_ratio * biggest_face.height) * height)
            x2 = int((biggest_face.xmin + biggest_face.width * (1 + margin_ratio)) * width)
            y2 = int((biggest_face.ymin + biggest_face.height * (1 + margin_ratio)) * height)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.shape[0] < min_face_size or face_crop.shape[1] < min_face_size:
                print(f"😢 얼굴 너무 작음: {input_path}")
                continue

            resized_face = cv2.resize(face_crop, face_size)

            # 배경 제거
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            result = segmentor.process(rgb_face)
            mask = result.segmentation_mask

            # 부드러운 마스크
            blurred_mask = cv2.GaussianBlur(mask, (25, 25), 0)
            alpha = np.expand_dims(blurred_mask, axis=-1)

            # 흰색 배경
            white_bg = np.full_like(resized_face, 255, dtype=np.uint8)

            # 부드러운 합성
            output_img = (alpha * resized_face + (1 - alpha) * white_bg).astype(np.uint8)

            # 저장
            cv2.imwrite(output_path, output_img)

# === 실행 ===
input_root = "test"
output_root = "test_face_mediapipe"

extract_faces_with_segmentation(input_root, output_root)
