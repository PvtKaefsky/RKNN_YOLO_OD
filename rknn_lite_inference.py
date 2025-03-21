import os, cv2, time, numpy as np
from utils import *
from rknnlite.api import RKNNLite

conf_thres = 0.1
iou_thres = 0.45
input_width = 1280
input_height = 736
model_name = 'yolov8n'
model_path = "./model"
config_path = "./config"
result_path = "./result"
image_path = "ac_all_1.jpg"
video_path = 0
video_inference = True
RKNN_MODEL = f'{model_path}/{model_name}-{input_width}-{input_height}.rknn'
CLASSES = ['WB MSW v3',
'Wiren Board 7 On',
'Fluid Sensor',
'Fan On',
'Red Button Disabled',
'Counter',
'Lamp',
'Wiren Board 7 Off',
'6-Channel Relay On',
'C16',
'MEGA MT On',
'Multi Channel Energy Meter On',
'WB MSW v3 Alarm',
'Red Button Enabled',
'Fan Off',
'Multi Channel Energy Meter Off',
'6-Channel Relay Off',
'MEGA MT Off']
isExist = os.path.exists(result_path)


if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    ret = rknn_lite.init_runtime()
    if video_inference == True:
        cap = cv2.VideoCapture(video_path)
        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            print('--> Running model for video inference')
            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            ret = rknn_lite.init_runtime()
            start = time.time()
            outputs = rknn_lite.inference(inputs=[image_3c])
            stop = time.time()
            fps = round(1/(stop-start), 2)
            outputs[0]=np.squeeze(outputs[0])
            outputs[0] = np.expand_dims(outputs[0], axis=0)
            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, shape = results
            vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            cv2.imshow("vis_img", vis_img)
            cv2.waitKey(10)
    else:
        image_3c = cv2.imread(image_path)
        image_4c, image_3c = preprocess(image_3c, input_height, input_width)
        ret = rknn_lite.init_runtime()
        start = time.time()
        outputs = rknn_lite.inference(inputs=[image_3c])
        stop = time.time()
        fps = round(1/(stop-start), 2)
        outputs[0]=np.squeeze(outputs[0])
        outputs[0] = np.expand_dims(outputs[0], axis=0)
        colorlist = gen_color(len(CLASSES))
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, shape = results
        if isinstance(boxes, np.ndarray):
            vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            print('--> Save inference result')
        else:
            print("No Detection result")
    print("RKNN inference finish")
    rknn_lite.release()
    cv2.destroyAllWindows()
