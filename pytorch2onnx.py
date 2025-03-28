import os, shutil
from ultralytics import YOLO

model_name = 'yolov8n'
input_width = 1280
input_height = 736
model_path = "./model"

isExist = os.path.exists(model_path)
if not isExist:
   os.makedirs(model_path)

model = YOLO(f"{model_name}.pt") 
model.export(format="onnx", imgsz=[input_width, input_height], opset=12)
os.rename(f"{model_name}.onnx", f"{model_name}-{input_width}-{input_height}.onnx")
shutil.move(f"{model_name}-{input_width}-{input_height}.onnx", f"./{model_path}/{model_name}-{input_width}-{input_height}.onnx")
shutil.move(f"{model_name}.pt", f"./{model_path}/{model_name}.pt")