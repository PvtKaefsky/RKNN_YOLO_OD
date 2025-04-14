import os
import json
import time
from datetime import datetime
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# ---------------- CONFIG ---------------- #
MODEL_NAME = "yolov8n"
INPUT_WIDTH, INPUT_HEIGHT = 640, 480
BATCH_SIZE = 1
PRECISION = "INT8"  # "FP16", "INT8"
WORKSPACE_GB = 10

ONNX_PATH = f"./model/{MODEL_NAME}-{INPUT_WIDTH}-{INPUT_HEIGHT}.onnx"
ENGINE_PATH = f"./model/{MODEL_NAME}-{INPUT_WIDTH}-{INPUT_HEIGHT}.engine"
CALIB_CACHE = f"./model/{MODEL_NAME}_calib.cache"
CALIB_DATASET_PATH = "./dataset"

CLASSES = {
    0: 'WB MSW v3', 1: 'Wiren Board 7 On', 2: 'Fluid Sensor', 3: 'Fan On',
    4: 'Red Button Disabled', 5: 'Counter', 6: 'Lamp', 7: 'Wiren Board 7 Off',
    8: '6-Channel Relay On', 9: 'C16', 10: 'MEGA MT On', 11: 'Multi Channel Energy Meter On',
    12: 'WB MSW v3 Alarm', 13: 'Red Button Enabled', 14: 'Fan Off',
    15: 'Multi Channel Energy Meter Off', 16: '6-Channel Relay Off', 17: 'MEGA MT Off'
}
# ---------------------------------------- #


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, cache_file=CALIB_CACHE):
        super().__init__()
        self.data = calibration_data
        self.cache_file = cache_file
        self.index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch_size(self):
        return self.data[0].shape[0]

    def get_batch(self, names):
        if self.index >= len(self.data):
            return None
        batch = np.ascontiguousarray(self.data[self.index])
        cuda.memcpy_htod(self.device_input, batch)
        self.index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def load_calibration_images(folder, max_imgs=32):
    print(f"Loading calibration images from {folder}...")
    img_data = []
    for idx, fname in enumerate(sorted(os.listdir(folder))):
        if idx >= max_imgs:
            break
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        img_data.append(img[np.newaxis, ...])  # (1, 3, H, W)

    if not img_data:
        raise RuntimeError("No valid calibration images found.")

    img_data = np.concatenate(img_data, axis=0)
    batches = [img_data[i:i+BATCH_SIZE] for i in range(0, len(img_data), BATCH_SIZE)]
    return batches


def build_trt_engine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX")

    print("\nNetwork IO description:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"[Input] {inp.name}, shape: {inp.shape}, dtype: {inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"[Output] {out.name}, shape: {out.shape}, dtype: {out.dtype}")

    calibrator = None

    if PRECISION.upper() == "FP16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 mode")

    elif PRECISION.upper() == "INT8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        calib_batches = load_calibration_images(CALIB_DATASET_PATH)
        calibrator = EntropyCalibrator(calib_batches)
        print(f"Using INT8 mode with {len(calib_batches)} batches")

        # Fallback to old-style API if needed
        try:
            config.int8_calibrator = calibrator  # some TRT builds still expect this
        except AttributeError:
            print("[Warning: config.int8_calibrator not available, proceeding with calibrator=None")

    print("Building engine...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    metadata = {
        "description": "YOLOv8 TensorRT model",
        "created": datetime.now().isoformat(),
        "version": "8.0.147",
        "task": "detect",
        "batch": BATCH_SIZE,
        "imgsz": [INPUT_WIDTH, INPUT_HEIGHT],
        "names": CLASSES
    }

    with open(ENGINE_PATH, "wb") as f:
        meta_json = json.dumps(metadata)
        f.write(len(meta_json).to_bytes(4, byteorder="little", signed=True))
        f.write(meta_json.encode())
        f.write(serialized_engine)

    print(f"Engine saved: {ENGINE_PATH}")
    print(f"Build time: {time.time() - t0:.2f} sec")


if __name__ == "__main__":
    build_trt_engine()
