from roboflow import Roboflow

rf = Roboflow(api_key="56TpUtANhNkiN2fYclyG")
project = rf.workspace("mahmoud-9wyf6").project("fire-n-smoke-detection")
version = project.version(1)
dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)
