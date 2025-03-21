import os, shutil, time, numpy as np, cv2
from utils import *
from ultralytics import YOLO

conf_thres = 0.25
iou_thres = 0.45
input_width = 1280
input_height = 736
model_name = 'yolov8n'
model_path = "./model"
config_path = "./config"
result_path = "./result"
image_path = "ac_all_1.jpg"
video_path = "ac_all_video.mp4"
video_inference = True
RKNN_MODEL = f'{model_name}-{input_width}-{input_height}.rknn'
CLASSES = ['WB MSW v3', 'Wiren Board 7 On', 'Fluid Sensor', 'Fan On', 'Red Button Disabled',
           'Counter', 'Lamp', 'Wiren Board 7 Off', '6-Channel Relay On', 'C16', 'MEGA MT On',
           'Multi Channel Energy Meter On', 'WB MSW v3 Alarm', 'Red Button Enabled', 'Fan Off',
           'Multi Channel Energy Meter Off', '6-Channel Relay Off', 'MEGA MT Off']

isExist = os.path.exists(model_path)
if not isExist:
   os.makedirs(model_path)

model = YOLO(f"{model_name}.pt") 
model.export(format="rknn", imgsz=[input_height, input_width], name="rk3588")

os.rename(f"{model_name}.onnx", f"{model_name}-{input_width}-{input_height}.onnx")
os.rename(f"{model_name}_rknn_model", f"{model_name}-{input_width}-{input_height}_rknn_model")
shutil.move(f"{model_name}-{input_width}-{input_height}.onnx", f"./{model_path}/{model_name}-{input_width}-{input_height}.onnx")

isExist = os.path.exists(f"./{model_path}/{model_name}-{input_width}-{input_height}_rknn_model")
if isExist:
    shutil.rmtree(f"./{model_path}/{model_name}-{input_width}-{input_height}_rknn_model")

shutil.move(f"{model_name}-{input_width}-{input_height}_rknn_model", f"./{model_path}")
shutil.move(f"{model_name}.pt", f"./{model_path}/{model_name}.pt")

if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)
    rknn_model = YOLO(f"./{model_path}/{model_name}-{input_width}-{input_height}_rknn_model")
    frames, loopTime, initTime = 0, time.time(), time.time()
    if video_inference == True:
        cap = cv2.VideoCapture(video_path)
        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            print('--> Running model for video inference')
            image_4c, image_3c, ratio, dwdh = preprocess(image_3c, input_height, input_width)
            outputs = rknn_model([image_3c])
            frames += 1
            outputs[0] = np.squeeze(outputs[0])
            outputs[0] = np.expand_dims(outputs[0], axis=0)
            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, ratio, dwdh, classes=len(CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, shape = results
            vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            cv2.imshow("vis_img", vis_img)
            cv2.waitKey(10)
            if frames % 30 == 0:
                print(f"Average FPS (last 30 frames): {30 / (time.time() - loopTime):.2f}")
                loopTime = time.time()
        avg_fps = frames / (time.time() - initTime)
        print(f"Overall Average FPS: {avg_fps:.2f}")
    else:
        image_3c = cv2.imread(image_path)
        image_4c, image_3c, ratio, dwdh = preprocess(image_3c, input_height, input_width)
        start = time.time()
        outputs = rknn_model([image_3c])
        stop = time.time()
        fps = round(1/(stop-start), 2)
        outputs[0]=np.squeeze(outputs[0])
        outputs[0] = np.expand_dims(outputs[0], axis=0)
        colorlist = gen_color(len(CLASSES))
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, ratio, dwdh, classes=len(CLASSES)) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, shape = results
        if isinstance(boxes, np.ndarray):
            vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            print('--> Save inference result')
        else:
            print("No Detection result")
    print("RKNN inference finish")
    cv2.destroyAllWindows()