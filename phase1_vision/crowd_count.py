import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, imgsz=480, persist=True, classes=[0], verbose=False)
    annotated = results[0].plot()

    person_count = len(results[0].boxes) if results[0].boxes is not None else 0

    cv2.putText(annotated, f"People count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AegisAI - Phase 2 Crowd Count", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
