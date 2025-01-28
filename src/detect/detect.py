import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(os.getpid() % 3)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from ultralytics import YOLO
import turbojpeg
import json
from PIL import Image
import torch

MIN_CONF=0.15

jpeg = turbojpeg.TurboJPEG()

vram_avail, vram_total = torch.cuda.mem_get_info()
if vram_avail < 6*(2**30):
    model = YOLO("./models/yolov8s_panoramax.pt")
    print("loading YOLO8s base model")
    model_name = 'yolo8s'
else:
    model = YOLO("./models/yolo11l_panoramax.pt")
    model_name = 'yolo11l'
names = ['sign','plate','face']


def iou(box1, box2):
    """ compute Intersection over Union (IoU) of two bbox
    """
    Aint = max(box1[0], box2[0])
    Bint = max(box1[1], box2[1])
    Cint = min(box1[2], box2[2])
    Dint = min(box1[3], box2[3])
    if Cint<Aint or Dint<Bint:
        return 0
    inter = (Cint-Aint) * (Dint-Bint)
    X = (box1[2]-box1[0]) * (box1[3]-box1[1])
    Y = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = X+Y-inter
    return inter/union


def detector(picture, cls=''):
    """Detect faces and licence plates in a single picture.

    Parameters
    ----------
    picture : tempfile
		Picture file

    Returns
    -------
    Bytes
        the blurred image
    """

    pid = os.getpid()

    # copy received JPEG picture to temporary file
    tmp = '/dev/shm/detect%s.jpg' % pid
    tmp_left = '/dev/shm/detect%sL.jpg' % pid
    tmp_right = '/dev/shm/detect%sR.jpg' % pid
    src = [tmp]
    result = []
    split = 0
    offset = []

    with open(tmp, 'w+b') as jpg:
        jpg.write(picture.file.read())

    # get picture details
    with open(tmp, 'rb') as jpg:
        width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(
            jpg.read())

    # detect on reduced image to detect large close-up objects
    try:
        results = model.predict(src, conf=MIN_CONF, imgsz=1024, half=True, verbose=False)
        result.append(results[0])
        offset.append(0)
    except:
        return None

    # detect with standard resolution
    try:
        results = model.predict(src, conf=MIN_CONF, imgsz=2048, half=True, verbose=False)
        result.append(results[0])
        offset.append(0)
    except:
        return None

    if width>=5760:
        # panoramic / 360° pictures
        if width >= height * 2:
            # split image in left and right parts to save VRAM
            split = int(width/2)
            with Image.open(tmp) as img_left:
                img_left.crop((0,0,split-1,height)).save(tmp_left)
            with Image.open(tmp) as img_right:
                img_right.crop((split,0,width,height)).save(tmp_right)
            src = [tmp_left, tmp_right]

        # detect again at higher resolution for smaller objects
        try:
            results = model.predict(source=src[0], conf=MIN_CONF, imgsz=min(int(width) >> 5 << 5,3840), half=True, verbose=False)
            result.append(results[0])
            offset.append(0)
            if len(src)>1:
                results = model.predict(source=src[1], conf=MIN_CONF, imgsz=min(int(width) >> 5 << 5,3840), half=True, verbose=False)
                result.append(results[0])
                offset.append(split)
        except:
            return None

    # prepare MCU crop rect list
    crop_rects = []

    # get MCU maximum size (2^n) = 8 or 16 pixels subsamplings
    hblock, vblock, sample = [(3, 3 ,'1x1'), (4, 3, '2x1'), (4, 4, '2x2'), (4, 4, '2x2'), (3, 4, '1x2')][jpeg_subsample]

    info = []
    for r in reversed(range(len(result))):
        for b in range(len(result[r].boxes)):
            obj = result[r].boxes[b]
            if cls !='' and not names[int(obj.cls)] in cls:
                continue
            box = obj.xywh
            box_l = int(offset[r] + box[0][0] - box[0][2] * 0.5) >> hblock << hblock
            box_t = int(box[0][1] - box[0][3] * 0.5) >> vblock << vblock
            box_w = int(box[0][2]) + (2 << hblock) >> hblock << hblock
            if names[int(obj.cls)] == 'sign':
                box_h = int(box[0][3] * 1.25 + (2 << vblock)) >> vblock << vblock
            else:
                box_h = int(box[0][3]) + (2 << vblock) >> vblock << vblock

            crop = [ max(0,box_l),
                    max(0,box_t),
                    min(box_w, width-max(0,box_l)),
                    min(box_h, height-max(0,box_t))]
            bbox = [int(offset[r] + obj.xyxy[0][0]),
                    int(obj.xyxy[0][1]),
                    int(offset[r] + obj.xyxy[0][2]),
                    int(obj.xyxy[0][3])]

            # remove overlaping detections
            for c in range(len(crop_rects)):
                if iou(bbox, info[c]['bbox']) > 0.33 or (crop[0] >= crop_rects[c][0]
                    and crop[1] >= crop_rects[c][1]
                    and crop[0]+crop[2] <= crop_rects[c][0]+crop_rects[c][2]
                    and crop[1]+crop[3] <= crop_rects[c][1]+crop_rects[c][3]):
                    # if new detection is smaller replace the previous one
                    if crop[2] * crop[3] < crop_rects[c][2] * crop_rects[c][3]:
                        crop_rects[c] =crop
                        info[c] = {
                            "class": names[int(obj.cls)],
                            "confidence": round(float(obj.conf),3),
                            "xywh": crop_rects[c],
                            "bbox": bbox
                            }
                    crop = None
                    break

            if crop:
                # collect info about blurred object to return to client
                crop_rects.append(crop)
                info.append({
                    "class": names[int(obj.cls)],
                    "confidence": round(float(obj.conf),3),
                    "xywh": crop_rects[-1],
                    "bbox": bbox
                    })

    return(json.dumps({'model': model_name, 'info': info, 'crop_rects': crop_rects}))
