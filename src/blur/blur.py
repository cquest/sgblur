import os, subprocess

from ultralytics import YOLO
import turbojpeg
from PIL import Image, ImageFilter

jpeg = turbojpeg.TurboJPEG()
model = YOLO("./models/yolov8s_panoramax.pt")
model.names[0] = 'sign'
model.names[1] = 'plate'
model.names[2] = 'face'

def blurPicture(picture):
    """Blurs a single picture using a given mask and returns blurred version.

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
    tmp = '/dev/shm/blur%s.jpg' % pid
    with open(tmp, 'wb') as jpg:
        jpg.write(picture.file.read())

    # call our detection model and dispatch threads on GPUs
    results = model.predict(source=tmp,
                            classes=[1, 2],
                            conf=0.1,
                            device=[pid % 2])
    result = results[0]

    with open(tmp, 'rb') as jpg:
        width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(jpg.read())

    if len(result.boxes) > 0:
        # prepare bounding boxes list
        crop_rects = []
        blocks = 4 # 16x16 pixels
        for obj in result.boxes:
            box = obj.xywh
            box_x = int(box[0][0]-(2 << (blocks-1))-box[0][2]/2)
            box_y = int(box[0][1]-(2 << (blocks-1))-box[0][3]/2)
            box_w = int(box[0][2]+(2 << blocks))
            box_h = int(box[0][3]+(2 << blocks))
            crop_rects.append([max(0, box_x >> blocks << blocks),
                               max(0, box_y >> blocks << blocks),
                               box_w >> blocks << blocks,
                               box_h >> blocks << blocks])

        # extract cropped jpeg data from boxes
        with open(tmp, 'rb') as jpg:
            crops = jpeg.crop_multiple(jpg.read(), crop_rects, background_luminance=0)

        # blur boxes and paste them onto original
        tmpcrop = '/dev/shm/crop%s.jpg' % os.getpid()
        for c in range(len(crops)):
            crop = open(tmpcrop,'wb')
            crop.write(crops[c])
            crop.close()
            # pillow based blurring
            img = Image.open(tmpcrop)
            boxblur = img.filter(ImageFilter.BoxBlur(32))
            boxblur.save(tmpcrop, subsampling=jpeg_subsample)
            # jpegtran "drop"
            subprocess.run('/bin/jpegtran -optimize -copy all -drop +%s+%s %s %s > %s' % (crop_rects[c][0], crop_rects[c][1], tmpcrop, tmp, tmp+'_tmp'), shell=True)
            os.replace(tmp+'_tmp', tmp)
        os.remove(tmpcrop)
    
    # return result (original image is no blur needed)
    with open(tmp, 'rb') as jpg:
        original = jpg.read()
    os.remove(tmp)
    return original
