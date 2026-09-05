import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=480, verbose=False)
    annotated = results[0].plot()

    for r in results:
        if r.keypoints is None or r.keypoints.conf is None:
            continue

        for kpts_conf in r.keypoints.conf:
            if len(kpts_conf) < 17:
                continue
            ls, rs, lh, rh = kpts_conf[5], kpts_conf[6], kpts_conf[11], kpts_conf[12]
            print(f"shoulder_conf=({ls:.2f},{rs:.2f}) hip_conf=({lh:.2f},{rh:.2f})")

    cv2.imshow("AegisAI - Debug Keypoint Confidence", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
