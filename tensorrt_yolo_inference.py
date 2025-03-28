import os
import time
import shutil
import numpy as np
import cv2
import tensorrt as trt
from ultralytics import YOLO
from utils import *

# Параметры
conf_thres = 0.5
iou_thres = 0.5
input_width = 1280
input_height = 736
model_name = 'yolov8n'
model_path = "./model"
result_path = "./result"
image_path = "ac_all_1.jpg"
video_path = "ac_all_video.mp4"
video_inference = True
precision = "FP16"  # Можно сменить на "INT8"

TRT_MODEL = f"{model_path}/{model_name}-{input_width}-{input_height}.engine"

CLASSES = [
    'WB MSW v3', 'Wiren Board 7 On', 'Fluid Sensor', 'Fan On', 'Red Button Disabled',
    'Counter', 'Lamp', 'Wiren Board 7 Off', '6-Channel Relay On', 'C16', 'MEGA MT On',
    'Multi Channel Energy Meter On', 'WB MSW v3 Alarm', 'Red Button Enabled', 'Fan Off',
    'Multi Channel Energy Meter Off', '6-Channel Relay Off', 'MEGA MT Off'
]

# # Создание папок
# os.makedirs(model_path, exist_ok=True)

# # === 1. Экспорт в TensorRT ===
# print("[INFO] Экспорт YOLOv8 в TensorRT...")
# model = YOLO(f"{model_name}.pt", imgsz=[input_width, input_height])

# # Прямой экспорт в TensorRT
# model.export(format="engine", imgsz=[input_width, input_height], half=(precision == "FP16"), int8=(precision == "INT8"))

# # Переименование и перемещение файла
# if not os.path.exists(f"{model_name}.engine"):
#     raise FileNotFoundError("Ошибка: TensorRT Engine не был создан!")

# shutil.move(TRT_MODEL, model_path)
# shutil.move(f"{model_name}.pt", f"{model_path}/{model_name}.pt")

# print(f"[INFO] TensorRT Engine сохранён в {TRT_MODEL}")

# === 2. Инференс TensorRT ===
if __name__ == '__main__':
    os.makedirs(result_path, exist_ok=True)

    # Загрузка TensorRT модели через Ultralytics API
    print("[INFO] Загрузка TensorRT Engine...")
    model = YOLO(TRT_MODEL)

    def infer(image):
        image_4c, image_3c = preprocess(image, input_width, input_height)
        start = time.time()
        outputs = model(image_3c)
        end = time.time()
        print(f"[INFO] Время инференса: {end - start:.3f} сек")
        outputs = np.array(outputs[0].boxes.data.cpu())  # Приведение к numpy
        return outputs, image_3c, image_4c

    # Видеоинференс
    if video_inference:
        cap = cv2.VideoCapture(video_path)
        frames, loopTime, initTime = 0, time.time(), time.time()

        while True:
            ret, image = cap.read()
            if not ret:
                break

            print("--> Видеоинференс...")
            outputs, image_3c, image_4c = infer(image)

            frames += 1
            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES))
            results = results[0]

            vis_img = vis_result(image_3c, results, colorlist, CLASSES, result_path)
            cv2.imshow("vis_img", vis_img)
            cv2.waitKey(10)

            if frames % 30 == 0:
                print(f"FPS (30 кадров): {30 / (time.time() - loopTime):.2f}")
                loopTime = time.time()

        avg_fps = frames / (time.time() - initTime)
        print(f"Средний FPS: {avg_fps:.2f}")

    # Инференс для изображения
    else:
        image = cv2.imread(image_path)
        outputs, image_3c, image_4c = infer(image)

        colorlist = gen_color(len(CLASSES))
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES))
        results = results[0]

        if isinstance(results[0], np.ndarray):
            vis_result(image_3c, results, colorlist, CLASSES, result_path)
            print("--> Сохранение результата")
        else:
            print("Объекты не найдены.")

    print("TensorRT инференс завершен.")
    cv2.destroyAllWindows()
