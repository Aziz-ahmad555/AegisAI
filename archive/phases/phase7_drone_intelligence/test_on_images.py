from ultralytics import YOLO
import os

model = YOLO(r"runs\detect\train\weights\best.pt")

test_dir = r"Sard-4\test\images"
test_images = os.listdir(test_dir)[:5]  # just test on first 5 images

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    results = model(img_path, imgsz=416, conf=0.4, verbose=False)
    annotated = results[0].plot()

    output_name = f"test_result_{img_name}"
    results[0].save(filename=output_name)
    print(f"Processed {img_name} -> saved as {output_name}")
