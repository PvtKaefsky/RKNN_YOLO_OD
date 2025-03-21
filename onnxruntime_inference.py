import os, time, random, numpy as np, cv2, onnxruntime
from utils import *

conf_thres = 0.25
iou_thres = 0.45
input_width = 1280
input_height = 736
result_path = "./result"
image_path = "ac_all_1.jpg"
video_path = "ac_all_video.mp4"
# video_path = 0
model_name = 'yolov8n'
model_path = "./model"
ONNX_MODEL = f"{model_path}/{model_name}-{input_width}-{input_height}.onnx"
video_inference = True
CLASSES = ['WB MSW v3', 'Wiren Board 7 On', 'Fluid Sensor', 'Fan On', 'Red Button Disabled',
           'Counter', 'Lamp', 'Wiren Board 7 Off', '6-Channel Relay On', 'C16', 'MEGA MT On',
           'Multi Channel Energy Meter On', 'WB MSW v3 Alarm', 'Red Button Enabled', 'Fan Off',
           'Multi Channel Energy Meter Off', '6-Channel Relay Off', 'MEGA MT Off']

sess = onnxruntime.InferenceSession(ONNX_MODEL)
input_list = [sess.get_inputs()[i].name for i in range (len(sess.get_outputs()))]
output_list = [sess.get_outputs()[i].name for i in range (len(sess.get_outputs()))]
isExist = os.path.exists(result_path)
if not isExist:
    os.makedirs(result_path)

if video_inference == True:
    cap = cv2.VideoCapture(video_path)
    frames, loopTime, initTime = 0, time.time(), time.time()
    while(True):
        ret, image_3c = cap.read()
        if not ret:
            break
        print('--> Running model for video inference')
        image_4c, image_3c = preprocess_onnx(image_3c, input_width, input_height)
        outputs = sess.run(output_list, {sess.get_inputs()[0].name: image_4c.astype(np.float32)})
        frames += 1
        colorlist = gen_color(len(CLASSES))
        results = postprocess_onnx(outputs, image_4c, image_3c, conf_thres, iou_thres) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, shape = results
        vis_img = vis_result(image_3c, results, colorlist, CLASSES, result_path)
        cv2.imshow("vis_img", vis_img)
        cv2.waitKey(10)
        if frames % 30 == 0:
            print(f"Average FPS (last 30 frames): {30 / (time.time() - loopTime):.2f}")
            loopTime = time.time()
    avg_fps = frames / (time.time() - initTime)
    print(f"Overall Average FPS: {avg_fps:.2f}")

else:
    image_3c = cv2.imread(image_path)
    image_4c, image_3c = preprocess_onnx(image_3c, input_width, input_height)
    outputs = sess.run(output_list, {sess.get_inputs()[0].name: image_4c.astype(np.float32)})
    colorlist = gen_color(len(CLASSES))
    results = postprocess_onnx(outputs, image_4c, image_3c, conf_thres, iou_thres) ##[box,mask,shape]
    results = results[0]              ## batch=1
    boxes, shape = results
    vis_img = vis_result(image_3c, results, colorlist, CLASSES, result_path)

print("ONNX inference finish")
cv2.destroyAllWindows()