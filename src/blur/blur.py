import os, subprocess

from ultralytics import YOLO
import turbojpeg
from PIL import Image, ImageFilter, ImageOps
import hashlib, pathlib, time
import exifread
import json, uuid
import torch

jpeg = turbojpeg.TurboJPEG()
model = YOLO("./models/yolov8s_panoramax.pt")
model.names[0] = 'sign'
model.names[1] = 'plate'
model.names[2] = 'face'

crop_save_dir = '/data/crops'

def blurPicture(picture, keep):
    """Blurs a single picture by detecting faces and licence plates.

    Parameters
    ----------
    picture : tempfile
		Picture file
    keep : int
        1 to keep blurred part to allow deblur

    Returns
    -------
    Bytes
        the blurred image
    """

    pid = os.getpid()
    if 'SGBLUR_GPUS' in os.environ:
        gpu = pid % int(os.environ['SGBLUR_GPUS'])
    else:
        gpu = pid % torch.cuda.device_count()

    # copy received JPEG picture to temporary file
    tmp = '/dev/shm/blur%s.jpg' % pid
    tmpcrop = '/dev/shm/crop%s.jpg' % pid

    with open(tmp, 'w+b') as jpg:
        jpg.write(picture.file.read())
        jpg.seek(0)
        tags = exifread.process_file(jpg, details=False)
    print("original", os.path.getsize(tmp))

    # solve image orientation
    if 'Image Orientation' in tags:
        if 'normal' not in str(tags['Image Orientation']):
            subprocess.run('exiftran -a %s -o %s' % (tmp, tmp+'_tmp'), shell=True)
            print("after exiftran", os.path.getsize(tmp+'_tmp'))
            os.replace(tmp+'_tmp', tmp)

    # get picture details
    with open(tmp, 'rb') as jpg:
        width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(
            jpg.read())
        jpg.seek(0)

        if width>=3840:
            # call our detection model and dispatch threads on GPUs
            results = model.predict(source=tmp,
                                    conf=0.05,
                                    imgsz=min(int(width) >> 5 << 5,8192),
                                    device=[gpu])
            result = [results[0]]
            offset = [[0,0]]
        else:
            # call our detection model and dispatch threads on GPUs
            try:
                results = model.predict(source=tmp,
                                        conf=0.05,
                                        device=[gpu])
            except:
                return None,None
            result = [results[0]]
            offset = [[0,0]]

    info = []
    salt = None

    # prepare bounding boxes list
    crop_rects = []

    # get MCU maximum size (2^n) = 8 or 16 pixels subsamplings
    hblock, vblock, sample = [(3, 3 ,'1x1'), (4, 3, '2x1'), (4, 4, '2x2'), (4, 4, '2x2'), (3, 4, '1x2')][jpeg_subsample]

    blurred = False
    for r in range(len(result)):
        for b in range(len(result[r].boxes)):
            obj = result[r].boxes[b]
            box = obj.xywh
            box_x = int(offset[r][0] + box[0][0] - box[0][2]/2)
            box_y = int(offset[r][0] + box[0][1] - box[0][3]/2)
            box_w = int(box[0][2]+(2 << (hblock)))
            box_h = int(box[0][3]+(2 << (vblock)))
            crop_rects.append([max(0, box_x >> hblock << hblock),
                            max(0, box_y >> vblock << vblock),
                            min(box_w >> hblock << hblock,
                                width-max(0, box_x >> hblock << hblock)),
                            min(box_h >> vblock << vblock, height-max(0, box_y >> vblock << vblock))])

            # collect info about blurred object to return to client
            info.append({
                "class": model.names[int(obj.cls)],
                "confidence": round(float(obj.conf),3),
                "xywh": crop_rects[-1]
            })

    if len(crop_rects)>0:
        # extract cropped jpeg data from boxes to be blurred
        with open(tmp, 'rb') as jpg:
            crops = jpeg.crop_multiple(jpg.read(), crop_rects, background_luminance=0, copynone=True)

        # if face or plate, blur boxes and paste them onto original
        for c in range(len(crops)):
            if info[c]['class'] == 'sign':
                continue
            blurred = True
            crop = open(tmpcrop,'wb')
            crop.write(crops[c])
            crop.close()
            # pillow based blurring
            img = Image.open(tmpcrop)
            radius = max(int(max(img.width, img.height)/12) >> 3 << 3, 8)
            # pixelate first
            reduced = ImageOps.scale(img, 1/radius, resample=0)
            pixelated = ImageOps.scale(reduced, radius, resample=0)
            # and blur
            boxblur = pixelated.filter(ImageFilter.BoxBlur(radius))
            boxblur.save(tmpcrop, subsampling=jpeg_subsample)
            subprocess.run('djpeg %s | cjpeg -sample %s -optimize -dct float -baseline -quality 90 -outfile %s' % (tmpcrop, sample, tmpcrop+'_tmp'), shell=True)
            os.replace(tmpcrop+'_tmp', tmpcrop)
            # jpegtran "drop"
            print( 'crop size', os.path.getsize(tmpcrop))
            p = subprocess.run('jpegtran -restart 64 -optimize -copy all -drop +%s+%s %s %s > %s' % (crop_rects[c][0], crop_rects[c][1], tmpcrop, tmp, tmp+'_tmp'), shell=True)
            print( 'after drop', os.path.getsize(tmp+'_tmp'))
            if p.returncode != 0 :
                print('crop XY: ',crop_rects[c][0], crop_rects[c][1])
                # problem with original JPEG... we try to recompress it
                subprocess.run('djpeg %s | cjpeg -optimize -smooth 10 -dct float -baseline -quality 90 -outfile %s' % (tmp, tmp+'_tmp'), shell=True)
                print('after recompressing original', os.path.getsize(tmp+'_tmp'))
                os.replace(tmp+'_tmp', tmp)
                subprocess.run('jpegtran -optimize -copy all -drop +%s+%s %s %s > %s' % (crop_rects[c][0], crop_rects[c][1], tmpcrop, tmp, tmp+'_tmp'), shell=True)
            os.replace(tmp+'_tmp', tmp)

        # save detected objects data in JPEG comment at end of file
        with open(tmp, 'r+b') as jpg:
            jpg.seek(0, os.SEEK_END)
            jpg.seek(-2, os.SEEK_CUR)
            jpg.write(b'\xFF\xFE')
            jpg.write(len(str(info)+'  ').to_bytes(2, 'big'))
            jpg.write(str(info).encode())
            jpg.write(b'\xFF\xD9')

        # keep potential false positive and road signs original parts hashed
        if crop_save_dir != '':
            salt = str(uuid.uuid4())
            for c in range(len(crops)):
                if ((keep == '1' and info[c]['confidence'] < 0.5 and info[c]['class'] in ['face', 'plate'])
                        or (info[c]['confidence'] > 0.2 and info[c]['class'] == 'sign')):
                    h = hashlib.sha256()
                    h.update(((salt if not info[c]['class'] == 'sign' else str(info))+str(info[c])).encode())
                    cropname = h.hexdigest()+'.jpg'
                    dirname = crop_save_dir+'/'+info[c]['class']+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'
                    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
                    with open(dirname+cropname, 'wb') as crop:
                        crop.write(crops[c])
                    # round ctime/mtime to midnight
                    daytime = int(time.time()) - int(time.time()) % 86400
                    os.utime(dirname+cropname, (daytime, daytime))
            info = { 'info': info, 'salt': salt }

        if blurred:
            # regenerate EXIF thumbnail
            before_thumb = os.path.getsize(tmp)
            subprocess.run('exiftran -g -o %s %s' % (tmp+'_tmp', tmp), shell=True)
            os.replace(tmp+'_tmp', tmp)
            print("after thumbnail", os.path.getsize(tmp), (100*(os.path.getsize(tmp)-before_thumb)/before_thumb))

    # return result (original image if no blur needed)
    with open(tmp, 'rb') as jpg:
        original = jpg.read()
    
    try:
        os.remove(tmp)
        os.remove(tmpcrop)
    except:
        pass

    return original, info


def deblurPicture(picture, idx, salt):
    """Un-blur a part of a previously blurred picture by restoring the original saved part.

    Parameters
    ----------
    picture : tempfile
		Picture file
    idx : int
        Index in list of blurred parts (starts at 0)
    salt: str
        salt to compute hashed filename

    Returns
    -------
    Bytes
        the "deblurred" image if OK else None
    """

    try:
        pid = os.getpid()
        # copy received JPEG picture to temporary file
        tmp = '/dev/shm/deblur%s.jpg' % pid

        with open(tmp, 'w+b') as jpg:
            jpg.write(picture.file.read())

        # get JPEG comment to retrieve detected objects
        com = subprocess.run('rdjpgcom %s' % tmp, shell=True, capture_output=True)
        i = json.loads(com.stdout.decode().replace('\'','"'))

        # do not allow deblur with high confidence detected objects
        if i[idx]['conf']>0.5:
            return None

        # compute hashed filename containing original picture part
        h = hashlib.sha256()
        h.update((salt+str(i[idx])).encode())
        cropname = h.hexdigest()+'.jpg'
        cropdir = crop_save_dir+'/'+i[idx]['class']+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'

        # "drop" original picture part into provided picture
        crop_rect = i[idx]['xywh']
        subprocess.run('jpegtran -optimize -copy all -drop +%s+%s %s %s > %s' % (crop_rect[0], crop_rect[1], cropdir+cropname, tmp, tmp+'_tmp'), shell=True, capture_output=True)
        os.replace(tmp+'_tmp', tmp)

        with open(tmp, 'rb') as jpg:
            deblurred = jpg.read()
        
        os.remove(tmp)

        print('deblur: ',i[idx], cropname)
        return deblurred
    except:
        return None

