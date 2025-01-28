import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(os.getpid() % 3)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from ultralytics import YOLO
import turbojpeg
import json
from PIL import Image
import torch

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

    # call our detection model on reduced image to detect large close-up objects
    try:
        results = model.predict(src, conf=0.20, imgsz=1024, half=True, verbose=False)
        result.append(results[0])
        offset.append(0)
        if split>0:
            result.append(results[1])
            offset.append(split)
    except:
        return None

    # detect again with standard resolution
    try:
        results = model.predict(src, conf=0.20, imgsz=2048, half=True, verbose=False)
        result.append(results[0])
        offset.append(0)
        if split>0:
            result.append(results[1])
            offset.append(split)
    except:
        return None

    if width>=5760:
        # panoramic / 360° pictures
        if width >= height * 2:
            # split image in left and right parts
            split = int(width/2)
            with Image.open(tmp) as img_left:
                img_left.crop((0,0,split-1,height)).save(tmp_left)
            with Image.open(tmp) as img_right:
                img_right.crop((split,0,width,height)).save(tmp_right)
            src = [tmp_left, tmp_right]

        # detect again at higher resolution for smaller objects
        try:
            results = model.predict(source=src[0], conf=0.20, imgsz=min(int(width) >> 5 << 5,3840), half=True, verbose=False)
            result.append(results[0])
            offset.append(0)
            if split>0:
                results = model.predict(source=src[1], conf=0.20, imgsz=min(int(width) >> 5 << 5,3840), half=True, verbose=False)
                result.append(results[0])
                offset.append(split)
        except:
            return None

    # prepare bounding boxes list
    crop_rects = []

    # get MCU maximum size (2^n) = 8 or 16 pixels subsamplings
    hblock, vblock, sample = [(3, 3 ,'1x1'), (4, 3, '2x1'), (4, 4, '2x2'), (4, 4, '2x2'), (3, 4, '1x2')][jpeg_subsample]

    info = []
    bbox = []
    for r in range(len(result)):
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

            # remove overlaping detections
            crop = [ max(0,box_l),
                    max(0,box_t),
                    min(box_w, width-max(0,box_l)),
                    min(box_h, height-max(0,box_t))]
            for c in range(len(crop_rects)):
                if (crop[0] >= crop_rects[c][0]
                    and crop[1] >= crop_rects[c][1]
                    and crop[0]+crop[2] <= crop_rects[c][0]+crop_rects[c][2]
                    and crop[1]+crop[3] <= crop_rects[c][1]+crop_rects[c][3]):
                    crop = None
                    break
            if crop:
                crop_rects.append(crop)
                # collect info about blurred object to return to client
                info.append({
                    "class": names[int(obj.cls)],
                    "confidence": round(float(obj.conf),3),
                    "xywh": crop_rects[-1]
                })
                bbox.append({   'top': int(box[0][1]),
                                'left': int(offset[r]+box[0][0]),
                                'bottom': int(box[0][1]+box[0][3]),
                                'right': int(offset[r]+box[0][0]+box[0][2]),
                                'cls': names[int(obj.cls)],
                                'conf': round(float(obj.conf),3)
                            })

    return(json.dumps({'info':info, 'crop_rects': crop_rects, 'bbox': bbox}))
