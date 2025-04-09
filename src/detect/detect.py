import os

if 'SGBLUR_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(os.getpid() % int(os.environ['SGBLUR_DEVICES']))

from ultralytics import YOLO
import turbojpeg
import json, time, gc
from PIL import Image
import torch

MIN_CONF=0.15
VERBOSE=False
HALF=True

TIMING=False

jpeg = turbojpeg.TurboJPEG()

vram_avail, vram_total = torch.cuda.mem_get_info()
if vram_avail < 6*(2**30):
    model = YOLO("./models/yolov8s_panoramax.pt")
    print("loading YOLO8s base model")
    model_name = 'yolo8s'
else:
    model = YOLO("./models/yolo11m_panoramax.pt")
    model_name = 'yolo11m'
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


def timing(msg=''):
    if TIMING:
        print('detect:', round(time.time()-start,3), msg)

def vram_free():
    torch.cuda.empty_cache()
    gc.collect()

def model_detect(model, img, imgsz=1024, vram_need=1):
    results = None
    while not results:
        torch.cuda.empty_cache()
        vram_avail, vram_total = torch.cuda.mem_get_info()
        while vram_avail >> 30 < vram_need:
            print('wait for vram', vram_need, vram_avail >> 30)
            torch.cuda.empty_cache()
            time.sleep((time.time() % 1/5)) # random wait 0-0.2s
            vram_avail, vram_total = torch.cuda.mem_get_info()

        try:
            results = model.predict(img, conf=MIN_CONF, imgsz=imgsz, half=HALF, verbose=VERBOSE)
        except:
            pass
    vram_free()
    return results


def detector(picture, cls=''):
    """Detect faces and licence plates in a single picture.

    Parameters
    ----------
    picture : tempfile
		Picture file

    Returns
    -------
    Bytes
        detection result as json
    """

    global start
    start = time.time()
    timing('detect start')
    pid = os.getpid()

    # copy received JPEG picture to temporary file
    tmp = '/dev/shm/detect%s.jpg' % pid
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
    img = Image.open(tmp)
    src = [img]

    timing('detect S')
    results = model_detect(model, img, imgsz=1024, vram_need=1)
    result.append(results[0])
    offset.append([0,0])

    # detect with standard resolution
    timing('detect L')
    results = model_detect(model, img, imgsz=2048, vram_need=2)
    result.append(results[0])
    offset.append([0,0])

    if width>=5760:
        timing('split 360')
        # panoramic / 360° pictures
        if width >= height * 2:
            # split image in left and right parts to save VRAM
            split = int(width/2)
            height_offset = height/4
            src = [ img.crop((0,height_offset,split-1,height*3/4)),
                    img.crop((split,height_offset,width,height*3/4)) ]
        else:
            height_offset = 0

        timing('detect XL')
        # detect again at higher resolution for smaller objects
        off = 0
        for i in src:
            results = model_detect(model, i, imgsz=min(int(width) >> 5 << 5,4096), vram_need=3)
            result.append(results[0])
            offset.append([off,height_offset])
            off += split
        timing('detect XL end')

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
            box_l = int(offset[r][0] + box[0][0] - box[0][2] * 0.5) >> hblock << hblock
            box_t = int(offset[r][1] + box[0][1] - box[0][3] * 0.5) >> vblock << vblock
            box_w = int(box[0][2]) + (2 << hblock) >> hblock << hblock
            if names[int(obj.cls)] == 'sign':
                box_h = int(box[0][3] * 1.25 + (2 << vblock)) >> vblock << vblock
            else:
                box_h = int(box[0][3]) + (2 << vblock) >> vblock << vblock

            crop = [ max(0,box_l),
                    max(0,box_t),
                    min(box_w, width-max(0,box_l)),
                    min(box_h, height-max(0,box_t))]
            bbox = [int(offset[r][0] + obj.xyxy[0][0]),
                    int(offset[r][1] + obj.xyxy[0][1]),
                    int(offset[r][0] + obj.xyxy[0][2]),
                    int(offset[r][1] + obj.xyxy[0][3])]

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

	# For some reason garbage collection does not run automatically after
	# a call to an AI model, so it must be done explicitely
    vram_free()

    timing('detect finished')
    print('%s detections in %s Mpx in %ss' % (len(crop_rects), int(width*height/1000000), round(time.time()-start,1)))
    return(json.dumps({'model': model_name, 'info': info, 'crop_rects': crop_rects}))
