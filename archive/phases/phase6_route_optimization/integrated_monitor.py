import cv2
import time
from ultralytics import YOLO
from building_graph import build_building_graph, visualize_graph
from route_finder import find_evacuation_route

# Map camera "zones" to the graph edges they monitor.
# In a real system, each physical camera would be tied to a specific
# location; here we simulate one camera watching the CorridorA <-> ExitEmergency route.
ZONE_TO_EDGES = {
    "camera_corridor_a": [("CorridorA", "ExitEmergency"), ("CorridorA", "CorridorB")],
}

FIRE_CONF_THRESHOLD = 0.55

def get_camera_fire_status(model, frame):
    results = model(frame, imgsz=416, conf=FIRE_CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    fire_detected = False
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name in ("Fire", "Smoke"):
                fire_detected = True

    return fire_detected, annotated


if __name__ == "__main__":
    model = YOLO("fire_smoke_model.pt")
    G = build_building_graph()
    exits = ["ExitMain", "ExitEmergency"]
    start_room = "Room101"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    print("AegisAI Phase 6 - Live integrated route optimization")
    print("Camera is monitoring 'camera_corridor_a' zone.")
    print("Show a fire/smoke image to the camera to see the route auto-reroute.")
    print("Press 'q' to quit.\n")

    last_status = None
    last_check = 0
    check_interval = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fire_detected, annotated = get_camera_fire_status(model, frame)

        now = time.time()
        if now - last_check >= check_interval:
            last_check = now

            blocked_edges = []
            if fire_detected:
                blocked_edges = ZONE_TO_EDGES["camera_corridor_a"]

            path, length, exit_used = find_evacuation_route(G, start_room, exits, blocked_edges=blocked_edges)

            status = (fire_detected, tuple(path) if path else None)
            if status != last_status:
                last_status = status
                if fire_detected:
                    print(f"[FIRE DETECTED in corridor A zone] Rerouting...")
                else:
                    print(f"[Zone clear] Normal routing.")

                if path:
                    print(f"  Route: {' -> '.join(path)} (distance: {length}, exit: {exit_used})\n")
                else:
                    print(f"  WARNING: No safe route found from {start_room}!\n")

        label = "FIRE DETECTED - REROUTING" if fire_detected else "Zone clear - normal routing"
        color = (0, 0, 255) if fire_detected else (0, 255, 0)
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("AegisAI - Phase 6 Integrated Route Monitor", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save a final snapshot of the graph reflecting last known state
    final_blocked = ZONE_TO_EDGES["camera_corridor_a"] if last_status and last_status[0] else []
    final_path, _, _ = find_evacuation_route(G, start_room, exits, blocked_edges=final_blocked)
    visualize_graph(G, path=final_path, blocked_edges=final_blocked, save_path="final_route_state.png")
