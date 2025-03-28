import os, shutil, time, numpy as np, cv2
from utils import *
from rknn.api import RKNN

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
video_inference = False
RKNN_MODEL = f'{model_name}-{input_width}-{input_height}.rknn'
CLASSES = ['WB MSW v3', 'Wiren Board 7 On', 'Fluid Sensor', 'Fan On', 'Red Button Disabled',
           'Counter', 'Lamp', 'Wiren Board 7 Off', '6-Channel Relay On', 'C16', 'MEGA MT On',
           'Multi Channel Energy Meter On', 'WB MSW v3 Alarm', 'Red Button Enabled', 'Fan Off',
           'Multi Channel Energy Meter Off', '6-Channel Relay Off', 'MEGA MT Off']

if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Build model
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input=f'{config_path}/{model_name}-{input_width}-{input_height}.model',
                                         data_input=f'{config_path}/{model_name}-{input_width}-{input_height}.data',
                                         model_quantization_cfg=f'{config_path}/{model_name}-{input_width}-{input_height}.quantization.cfg')
    
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    print('--> Move RKNN file into model folder')
    shutil.move(RKNN_MODEL, f"{model_path}/{RKNN_MODEL}")

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

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
            print('--> Running model for video inference')
            outputs = rknn.inference(inputs=[image_3c])
            frames += 1
            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
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
        # Preprocess input image
        image_3c = cv2.imread(image_path)
        image_4c, image_3c = preprocess(image_3c, input_width, input_height)
        print('--> Running model for image inference')
        print(image_3c.shape)
        outputs = rknn.inference(inputs=[image_3c])
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
    rknn.release()
    cv2.destroyAllWindows()