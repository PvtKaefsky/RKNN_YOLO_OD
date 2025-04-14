import json
import os
import threading
import time
import numpy as np
import cv2
import torch
from torch import nn
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import OrderedDict, namedtuple
from tensorrt_pool import TensorRTPool
from utils import preprocess, postprocess, vis_result, gen_color

# ------------------- CONFIG ------------------- #
model_name = "yolov8n"
video_path = "./ac_all_video.mp4"
result_path = "./result"
input_width, input_height = 640, 480
conf_thres, iou_thres = 0.25, 0.45
engine_path = f"./model/{model_name}-{input_width}-{input_height}.engine"
device = torch.device('cuda:0')
infer_every_n = 4  # Inference every N frames

CLASSES = [
    'WB MSW v3', 'Wiren Board 7 On', 'Fluid Sensor', 'Fan On', 'Red Button Disabled',
    'Counter', 'Lamp', 'Wiren Board 7 Off', '6-Channel Relay On', 'C16', 'MEGA MT On',
    'Multi Channel Energy Meter On', 'WB MSW v3 Alarm', 'Red Button Enabled', 'Fan Off',
    'Multi Channel Energy Meter Off', '6-Channel Relay Off', 'MEGA MT Off'
]
# ------------------------------------------------


class TensorRTInferenceModel(nn.Module):
    def __init__(self, engine_path: str):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.INFO)
        self.device = torch.device("cuda")
        self.engine_path = engine_path
        self.lock = threading.Lock()
        self._load_engine()

    def _load_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            meta_len = int.from_bytes(f.read(4), byteorder="little")
            self.metadata = json.loads(f.read(meta_len).decode("utf-8"))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.bindings = OrderedDict()
        self.input_names, self.output_names = [], []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)[2]
                self.context.set_input_shape(name, profile_shape)
                shape = profile_shape
            else:
                self.output_names.append(name)
                shape = self.engine.get_tensor_shape(name)

            tensor = torch.empty(tuple(shape), dtype=torch.float32, device=self.device)
            self.bindings[name] = Binding(name, dtype, shape, tensor, int(tensor.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def forward(self, image: torch.Tensor) -> list:
        image = image.to(self.device)
        input_name = self.input_names[0]
        with self.lock:
            self.context.set_input_shape(input_name, tuple(image.shape))
            self.binding_addrs[input_name] = int(image.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            outputs = [self.bindings[name].data.clone().detach().cpu().numpy()
                       for name in self.output_names]
        return outputs

    def to(self, device):
        return self

    def size(self):
        return os.path.getsize(self.engine_path)


def scale_boxes(boxes, from_shape, to_shape):
    scale_x = to_shape[1] / from_shape[1]
    scale_y = to_shape[0] / from_shape[0]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


def preprocess_for_inference(full_frame):
    resized = cv2.resize(full_frame, (input_width, input_height))
    img_4c, img_3c = preprocess(resized, input_width, input_height)
    tensor = torch.from_numpy(img_4c.astype(np.float32)).to(device)
    return tensor, img_3c, resized.shape[:2]


def main():
    cap = cv2.VideoCapture(video_path)
    pool = TensorRTPool(TensorRTInferenceModel, engine_path, num_workers=2)
    colorlist = gen_color(len(CLASSES))
    last_results = None

    frame_id = 0
    frames, loopTime, initTime = 0, time.time(), time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        original_shape = frame.shape[:2]

        if frame_id % infer_every_n == 0:
            img_tensor, img_3c, ratio, pad = preprocess(frame, input_width, input_height)
            outputs = pool.infer(img_tensor)
            results = postprocess(outputs, img_tensor, frame, conf_thres, iou_thres, classes=len(CLASSES), ratio_pad=(ratio, pad))
            if results:
                boxes, shape = results[0]
                if isinstance(boxes, np.ndarray):
                    scaled_boxes = scale_boxes(boxes.copy(), (input_height, input_width), original_shape)
                    last_results = (scaled_boxes, shape)

        if last_results:
            vis = vis_result(frame, results[0], colorlist, CLASSES, result_path)
        else:
            vis = frame

        cv2.imshow("Fast Inference", vis)
        cv2.waitKey(1)
        frame_id += 1
        frames += 1

        if frames % 30 == 0:
            print(f"FPS (30 frames): {30 / (time.time() - loopTime):.2f}")
            loopTime = time.time()

    avg_fps = frames / (time.time() - initTime)
    print(f"Average FPS: {avg_fps:.2f}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
