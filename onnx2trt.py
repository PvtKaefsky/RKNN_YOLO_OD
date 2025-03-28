import os
import time
import json
from datetime import datetime
import tensorrt as trt

task = "detect"
batch_size = 1

precision = "FP16"  # FP16, INT8
workspace = 1  # в ГБ

input_width = 1280
input_height = 736
model_name = 'yolov8n'
model_path = "./model"
config_path = "./config"
ONNX_MODEL = f'{model_path}/{model_name}-{input_width}-{input_height}.onnx'
TensorRT_MODEL = f'{model_path}/{model_name}-{input_width}-{input_height}.engine'

CLASSES = {
    0: 'WB MSW v3',
    1: 'Wiren Board 7 On',
    2: 'Fluid Sensor',
    3: 'Fan On',
    4: 'Red Button Disabled',
    5: 'Counter',
    6: 'Lamp',
    7: 'Wiren Board 7 Off',
    8: '6-Channel Relay On',
    9: 'C16',
    10: 'MEGA MT On',
    11: 'Multi Channel Energy Meter On',
    12: 'WB MSW v3 Alarm',
    13: 'Red Button Enabled',
    14: 'Fan Off',
    15: 'Multi Channel Energy Meter Off',
    16: '6-Channel Relay Off',
    17: 'MEGA MT Off'
}

if __name__ == '__main__':
    # Создание логгера
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    config = builder.create_builder_config()
    
    # Установка предела памяти
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30))  # ГБ -> байты

    parser = trt.OnnxParser(network, logger)

    # Проверка существования модели
    if not os.path.exists(ONNX_MODEL):
        raise FileNotFoundError(f'ONNX файл {ONNX_MODEL} не найден. Сначала запустите pytorch2onnx.py.')

    # Парсинг ONNX модели
    with open(ONNX_MODEL, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            raise RuntimeError(f'Ошибка загрузки ONNX файла: {ONNX_MODEL}')
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    print("Описание сети:")
    for inp in inputs:
        print(f"Вход '{inp.name}' с размерностью {inp.shape} и типом {inp.dtype}")
    for out in outputs:
        print(f"Выход '{out.name}' с размерностью {out.shape} и типом {out.dtype}")

    # Установка точности модели
    if builder.platform_has_fast_fp16 and precision == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif builder.platform_has_fast_int8 and precision == "INT8":
        config.set_flag(trt.BuilderFlag.INT8)

    start = time.time()

    metadata = {
        'description': "Ultralytics YOLOv8n model",
        'author': 'Ultralytics',
        'license': 'MIT License',
        'date': datetime.now().isoformat(),
        'version': "8.0.147",
        'stride': 32,
        'task': task,
        'batch': batch_size,
        'imgsz': [input_width, input_height],
        'names': CLASSES  # метаданные модели
    }

    # Новый метод компиляции модели в TensorRT 10.9
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Ошибка компиляции TensorRT Engine")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # Сохранение модели
    with open(TensorRT_MODEL, 'wb') as trt_file:
        meta_json = json.dumps(metadata)
        trt_file.write(len(meta_json).to_bytes(4, byteorder='little', signed=True))
        trt_file.write(meta_json.encode())
        trt_file.write(serialized_engine)

    print("Время выполнения: ", time.time() - start)
    print("TensorRT Engine сохранён")
