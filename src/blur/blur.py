import os, subprocess

from ultralytics import YOLO
import turbojpeg
from PIL import Image, ImageFilter
import hashlib, pathlib, time
import exifread

jpeg = turbojpeg.TurboJPEG()
model = YOLO("./models/yolov8s_panoramax.pt")
model.names[0] = 'sign'
model.names[1] = 'plate'
model.names[2] = 'face'

crop_save_dir = '/tmp/crops'

def blurPicture(picture):
    """Blurs a single picture by detecting faces and licence plates.

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
    with open(tmp, 'w+b') as jpg:
        jpg.write(picture.file.read())
        jpg.seek(0)
        tags = exifread.process_file(jpg, details=False)
    
    # solve image orientation
    if 'Image Orientation' in tags:
        if 'normal' not in str(tags['Image Orientation']):
            subprocess.run('exiftran -a %s -o %s' % (tmp, tmp+'_tmp'), shell=True)
            os.replace(tmp+'_tmp', tmp)

    # call our detection model and dispatch threads on GPUs
    results = model.predict(source=tmp,
                            classes=[1, 2],
                            conf=0.05,
                            device=[pid % 2])
    result = results[0]

    info = []
    if len(result.boxes) > 0:
        with open(tmp, 'rb') as jpg:
            width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(
                jpg.read())

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

            # collect info about blurred object to return to client
            info.append({
                "class": model.names[int(obj.cls)],
                "confidence": round(float(obj.conf),3),
                "xywh": crop_rects[-1]
            })


        # extract cropped jpeg data from boxes to be blurred
        with open(tmp, 'rb') as jpg:
            crops = jpeg.crop_multiple(jpg.read(), crop_rects, background_luminance=0, copynone=True)

        # blur boxes and paste them onto original
        tmpcrop = '/dev/shm/crop%s.jpg' % os.getpid()
        for c in range(len(crops)):
            crop = open(tmpcrop,'wb')
            crop.write(crops[c])
            crop.close()
            # pillow based blurring
            img = Image.open(tmpcrop)
            radius = max(int(max(img.width, img.height)/16) >> 3 << 3, 8)
            boxblur = img.filter(ImageFilter.BoxBlur(radius))
            boxblur.save(tmpcrop, subsampling=jpeg_subsample)
            if jpeg_subsample == 4:
                # resample crop in case of subsampling mismatch (4:4:0)
                subprocess.run('/bin/djpeg %s | /bin/cjpeg -sample 1x2 > %s' % (tmpcrop, tmpcrop+'_tmp'), shell=True)
                os.replace(tmpcrop+'_tmp', tmpcrop)
            # jpegtran "drop"
            subprocess.run('/bin/jpegtran -optimize -copy all -drop +%s+%s %s %s > %s' % (crop_rects[c][0], crop_rects[c][1], tmpcrop, tmp, tmp+'_tmp'), shell=True)
            os.replace(tmp+'_tmp', tmp)
        os.remove(tmpcrop)

        # save blur data in JPEG comment at end of file
        with open(tmp, 'r+b') as jpg:
            jpg.seek(0, os.SEEK_END)
            jpg.seek(-2, os.SEEK_CUR)
            jpg.write(b'\xFF\xFE')
            jpg.write(len(str(info)+'  ').to_bytes(2, 'big'))
            jpg.write(str(info).encode())
            jpg.write(b'\xFF\xD9')

        # keep potential false positive original parts hashed
        if crop_save_dir != '':
            for c in range(len(crops)):
                if info[c]['confidence'] < 0.5:
                    h = hashlib.sha256()
                    h.update((str(info)+str(info[c])).encode())
                    cropname = h.hexdigest()+'.jpg'
                    dirname = crop_save_dir+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'
                    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
                    with open(dirname+cropname, 'wb') as crop:
                        crop.write(crops[c])
                    # round ctime/mtime to midnight
                    daytime = int(time.time()) - int(time.time()) % 86400
                    os.utime(dirname+cropname, (daytime, daytime))

    # return result (original image is no blur needed)
    with open(tmp, 'rb') as jpg:
        original = jpg.read()
    os.remove(tmp)
    return original, info
