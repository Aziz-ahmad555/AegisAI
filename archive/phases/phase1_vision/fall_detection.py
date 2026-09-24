import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

CONF_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=480, verbose=False)
    annotated = results[0].plot()

    for r in results:
        if r.keypoints is None or r.keypoints.conf is None:
            continue

        for kpts_xy, kpts_conf in zip(r.keypoints.xy, r.keypoints.conf):
            if len(kpts_xy) < 17:
                continue

            needed = [5, 6, 11, 12]
            if any(kpts_conf[i] < CONF_THRESHOLD for i in needed):
                continue

            left_shoulder = kpts_xy[5]
            right_shoulder = kpts_xy[6]
            left_hip = kpts_xy[11]
            right_hip = kpts_xy[12]

            shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_x = (left_hip[0] + right_hip[0]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2

            dx = abs(hip_x - shoulder_x)
            dy = abs(hip_y - shoulder_y)

            # print raw numbers so we can see what's happening
            print(f"dx={dx:.1f} dy={dy:.1f}")

            # if the torso is more horizontal than vertical -> likely lying down
            if dx > dy:
                cv2.putText(annotated, "POSSIBLE FALL DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("AegisAI - Phase 2 Fall Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
