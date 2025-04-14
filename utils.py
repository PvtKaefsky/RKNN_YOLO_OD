import os, time, numpy as np, cv2

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        ids = np.where(iou <= threshold)[0]
        order = order[ids + 1]
    return keep

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,
):
    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    multi_label &= nc > 1

    t = time.time()
    output = [np.zeros((0,6 + nc))] * bs

    for xi, x in enumerate(prediction):
        x = np.transpose(x,[1, 0])
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls, mask = np.split(x, [4, 4+nc], axis=1)
        box_xyxy = xywh2xyxy(box)

        j = np.argmax(cls, axis=1)
        conf = cls[np.arange(j.shape[0]), j].reshape(-1,1)
        x = np.concatenate([box_xyxy, conf, j.reshape(-1,1), mask], axis=1)[conf.reshape(-1,) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        x = x[np.argsort(x[:, 4])[::-1][:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break
    return output

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def preprocess(image, input_width, input_height):
    image_3c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_3c, ratio, (dw, dh) = letterbox(image_3c, new_shape=(input_width, input_height), auto=False)
    image_4c = np.array(image_3c) / 255.0
    image_4c = np.transpose(image_4c, (2, 0, 1))  # HWC -> CHW
    image_4c = np.expand_dims(image_4c, axis=0).astype(np.float32)
    image_4c = np.ascontiguousarray(image_4c)
    return image_4c, image_3c, ratio, (dw, dh)

def postprocess(preds, img, orig_img, conf_thres, iou_thres, classes=None, ratio_pad=None):
    pred = non_max_suppression(preds[0], conf_thres, iou_thres, nc=classes)
    results = []
    for i, det in enumerate(pred):
        shape = orig_img.shape
        if not len(det):
            results.append([[], []])
            continue
        if ratio_pad is not None:
            ratio, (dw, dh) = ratio_pad
            det[:, [0, 2]] -= dw
            det[:, [1, 3]] -= dh
            det[:, :4] /= ratio[0]
            clip_boxes(det[:, :4], shape)
        else:
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], shape).round()
        results.append([det[:, :6], shape[:2]])
    return results

def gen_color(class_num):
    color_list = []
    np.random.seed(1)
    while True:
        a = list(map(int, np.random.choice(range(255),3)))
        if np.sum(a) == 0:
            continue
        color_list.append(a)
        if len(color_list) == class_num:
            break
    return color_list

def vis_result(image_bgr, results, colorlist, CLASSES, result_path):
    os.makedirs(result_path, exist_ok=True)
    boxes, _ = results
    vis_img = image_bgr.copy()
    cls_list = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls_id = box[:6]
        cls_id = int(cls_id)
        cls_list.append(cls_id)
        color = colorlist[cls_id]
        label = f"{CLASSES[cls_id]}: {round(conf, 2)}"
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis_img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for cls_id in set(cls_list):
        print(f"Found {cls_list.count(cls_id)} × {CLASSES[cls_id]}")
    cv2.imwrite(os.path.join(result_path, "origin_image.jpg"), image_bgr)
    cv2.imwrite(os.path.join(result_path, "visual_image.jpg"), vis_img)
    return vis_img