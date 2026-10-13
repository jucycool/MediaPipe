import cv2
import mediapipe as mp

# MediaPipe Face Detection 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 열기 (캠이 없으므로 face.mp4 사용)
cap = cv2.VideoCapture("face.mp4")

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("영상 스트림이 끝나거나 프레임을 읽지 못했습니다.")
            break

        # 좌우 반전 (셀카 뷰) 및 RGB 변환
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 얼굴 감지 수행
        results = face_detection.process(image_rgb)

        # 감지된 얼굴 표기 (네모 상자 및 눈/코/입 랜드마크)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # 결과 화면 표시
        cv2.imshow("MediaPipe Face Detection", image)

        # ESC 키로 종료
        if cv2.waitKey(5) & 0xFF == 27:
            print("종료합니다.")
            break

cap.release()
cv2.destroyAllWindows()