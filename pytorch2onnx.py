import os
import shutil
from ultralytics import YOLO

# ---------------- CONFIG ----------------
model_name = 'yolov8n'
input_width = 640
input_height = 480
model_dir = "./model"
input_size = [input_width, input_height]
onnx_opset = 12
# ----------------------------------------

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

pt_model_path = f"{model_name}.pt"
onnx_file_name = f"{model_name}-{input_width}-{input_height}.onnx"
onnx_output_path = os.path.join(model_dir, onnx_file_name)

print(f"Loading model from {pt_model_path}")
model = YOLO(pt_model_path)

print(f"Exporting to ONNX format...")
export_result = model.export(format="onnx", imgsz=input_size, opset=onnx_opset)

shutil.move(f"{model_name}.onnx", onnx_output_path)
shutil.move(pt_model_path, os.path.join(model_dir, f"{model_name}.pt"))

print(f"Model exported: {onnx_output_path}")