import os
import cv2
import json
import time
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import OrderedDict, namedtuple
from utils import *

# Параметры модели
conf_thres = 0.25
iou_thres = 0.45
input_width = 1280
input_height = 736
model_name = 'yolov8n'
model_path = "./model"
result_path = "./result"
image_path = "ac_all_1.jpg"
video_path = "ac_all_video.mp4"
video_inference = True
TensorRT_MODEL = f"{model_path}/{model_name}-{input_width}-{input_height}.engine"

CLASSES = [
    'WB MSW v3', 'Wiren Board 7 On', 'Fluid Sensor', 'Fan On', 'Red Button Disabled',
    'Counter', 'Lamp', 'Wiren Board 7 Off', '6-Channel Relay On', 'C16', 'MEGA MT On',
    'Multi Channel Energy Meter On', 'WB MSW v3 Alarm', 'Red Button Enabled', 'Fan Off',
    'Multi Channel Energy Meter Off', '6-Channel Relay Off', 'MEGA MT Off'
]

def main():
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    device = torch.device('cuda:0')
    # Read file
    with open(TensorRT_MODEL, 'rb') as f, trt.Runtime(logger) as runtime:
        meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
        metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
        model = runtime.deserialize_cuda_engine(f.read())  # read engine
    context = model.create_execution_context()
    bindings = OrderedDict()
    input_names = []
    output_names = []
    for i in range(model.num_io_tensors):
        name = model.get_tensor_name(i)
        dtype = trt.nptype(model.get_tensor_dtype(name))
        if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                dynamic = True
                context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[2]))
            if dtype == np.float16:
                fp16 = True
        else:  # output
            output_names.append(name)
        shape = tuple(context.get_tensor_shape(name))
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

    if video_inference == True:    
        cap = cv2.VideoCapture(video_path)
        frames, loopTime, initTime = 0, time.time(), time.time()
        while(True):
            ret, image_3c = cap.read()
            image_3c = cv2.flip(image_3c, 0)
            image_3c = cv2.flip(image_3c, 1)
            if not ret:
                break
            image_4c, image_3c = preprocess(image_3c, input_width, input_height)
            image_4c = image_4c.astype(np.float32)
            image_4c = torch.from_numpy(image_4c).to(device)
            binding_addrs['images'] = int(image_4c.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
            outputs = [bindings[x].data.cpu().numpy() for x in sorted(output_names)] # put the result from gpu to cpu and convert to numpy
            colorlist = gen_color(len(CLASSES))
            frames += 1
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, shape = results
            if isinstance(boxes, np.ndarray):
                vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
                cv2.imshow("vis_img", vis_img)
            else:
                print("No Detection result")
            cv2.waitKey(10)
            
            if frames % 30 == 0:
                    print(f"FPS (30 кадров): {30 / (time.time() - loopTime):.2f}")
                    loopTime = time.time()

        avg_fps = frames / (time.time() - initTime)
        print(f"Средний FPS: {avg_fps:.2f}")
        
    else:
        image_3c = cv2.imread(image_path)
        image_4c, image_3c = preprocess(image_3c, input_width, input_height)
        image_4c = image_4c.astype(np.float32)
        image_4c = torch.from_numpy(image_4c).to(device)
        binding_addrs['images'] = int(image_4c.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        outputs = [bindings[x].data.cpu().numpy() for x in sorted(output_names)] # put the result from gpu to cpu and convert to numpy
        colorlist = gen_color(len(CLASSES)) 
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, shape = results
        if isinstance(boxes, np.ndarray):
            vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            print('--> Save inference result')
        else:
            print("No segmentation result")
    print("TensorRT inference finish")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
