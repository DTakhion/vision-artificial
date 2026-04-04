# scripts/download_dataset.py
from dotenv import load_dotenv
import os
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("Falta ROBOFLOW_API_KEY en .env")

rf = Roboflow(api_key=api_key)

project = rf.workspace("digit-utaft").project("bar-code-uoyyp")
version = project.version(1)

# guardado
dataset = version.download(
    "yolov8",
    location="data/barcode_dataset_yolo"
)

print("Dataset descargado en data/barcode_dataset")