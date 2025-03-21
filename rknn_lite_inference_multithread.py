import os, time, numpy as np, cv2
from utils import *
from rknnpool import rknnPoolExecutor

# Параметры
conf_thres = 0.1
iou_thres = 0.45
input_width = 1280
input_height = 736
model_name = 'yolov8n'
model_path = "./model"
result_path = "./result"
video_path = "ac_all_video.mp4"
# video_path = 0
video_inference = True
RKNN_MODEL = f'{model_path}/{model_name}-{input_width}-{input_height}.rknn'
CLASSES = ['WB MSW v3', 'Wiren Board 7 On', 'Fluid Sensor', 'Fan On', 'Red Button Disabled',
           'Counter', 'Lamp', 'Wiren Board 7 Off', '6-Channel Relay On', 'C16', 'MEGA MT On',
           'Multi Channel Energy Meter On', 'WB MSW v3 Alarm', 'Red Button Enabled', 'Fan Off',
           'Multi Channel Energy Meter Off', '6-Channel Relay Off', 'MEGA MT Off']
TPEs = 3  # Число потоков

if not os.path.exists(result_path):
    os.makedirs(result_path)

def inference_func(rknn_lite, frame):
    image_4c, image_3c, ratio, dwdh = preprocess(frame, input_width, input_height)
    outputs = rknn_lite.inference(inputs=[image_3c])
    outputs[0] = np.squeeze(outputs[0])
    outputs[0] = np.expand_dims(outputs[0], axis=0)
    results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, ratio, dwdh, classes=len(CLASSES))
    results = results[0]
    boxes, shape = results
    colorlist = gen_color(len(CLASSES))
    vis_img = vis_result(image_3c, results, colorlist, CLASSES, result_path)
    return vis_img

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    pool = rknnPoolExecutor(rknnModel=RKNN_MODEL, TPEs=TPEs, func=inference_func)

    if cap.isOpened():
        for _ in range(TPEs + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                pool.release()
                exit(-1)
            pool.put(frame)

    frames, loopTime, initTime = 0, time.time(), time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)
        result_frame, flag = pool.get()
        if not flag:
            break
        frames += 1
        cv2.imshow('YOLOv8n Multithread Detection', result_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if frames % 30 == 0:
            print(f"Average FPS (last 30 frames): {30 / (time.time() - loopTime):.2f}")
            loopTime = time.time()

    avg_fps = frames / (time.time() - initTime)
    print(f"Overall Average FPS: {avg_fps:.2f}")

    cap.release()
    pool.release()
    cv2.destroyAllWindows()