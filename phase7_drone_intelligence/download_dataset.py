import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("phenikaax-tqlbw").project("sard-ijcjs")
version = project.version(4)
dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)
