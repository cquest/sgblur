import os, subprocess

from ultralytics import YOLO
import turbojpeg
from PIL import Image, ImageFilter, ImageOps
import hashlib, pathlib, time
import exifread
import json, uuid


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
    # copy received JPEG picture to temporary file
    tmp = '/dev/shm/blur%s.jpg' % pid
    tmpcrop = '/dev/shm/crop%s.jpg' % pid

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
    try:
        results = model.predict(source=tmp,
                                conf=0.05,
                                device=[pid % 2])
    except:
        return None,None

    result = results[0]

    info = []
    salt = None

    if len(result.boxes) > 0:
        with open(tmp, 'rb') as jpg:
            width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(
                jpg.read())

        # prepare bounding boxes list
        crop_rects = []

        # MCU maximum size (2^n) = 8 ou 16 pixels
        hblock, vblock = [(3, 3), (4, 3), (4, 4), (4, 4), (3, 4)][jpeg_subsample]

        for obj in result.boxes:
            box = obj.xywh

            box_x = int(box[0][0]-(2 << (hblock-1))-box[0][2]/2)
            box_y = int(box[0][1]-(2 << (vblock-1))-box[0][3]/2)
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

        # extract cropped jpeg data from boxes to be blurred
        with open(tmp, 'rb') as jpg:
            crops = jpeg.crop_multiple(jpg.read(), crop_rects, background_luminance=0, copynone=True)

        # if face or plate, blur boxes and paste them onto original
        for c in range(len(crops)):
            if info[c]['class'] == 'sign':
                continue
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
            boxblur.save(tmpcrop, subsampling=jpeg_subsample, quality=20)
            if jpeg_subsample == 4:
                # resample crop in case of subsampling mismatch (4:4:0)
                subprocess.run('djpeg %s | cjpeg -sample 1x2 -quality 20 -outfile %s' % (tmpcrop, tmpcrop+'_tmp'), shell=True)
                os.replace(tmpcrop+'_tmp', tmpcrop)
            # jpegtran "drop"
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
                        or (info[c]['confidence'] > 0.5 and info[c]['class'] == 'sign')):
                    h = hashlib.sha256()
                    h.update((salt+str(info[c])).encode())
                    cropname = h.hexdigest()+'.jpg'
                    dirname = crop_save_dir+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'
                    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
                    with open(dirname+cropname, 'wb') as crop:
                        crop.write(crops[c])
                    # round ctime/mtime to midnight
                    daytime = int(time.time()) - int(time.time()) % 86400
                    os.utime(dirname+cropname, (daytime, daytime))
            info = { 'info': info, 'salt': salt }

    # regenerate EXIF thumbnail
    subprocess.run('exiftran -g -i %s' % tmp, shell=True)

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
            jpg.seek(0)
            tags = exifread.process_file(jpg, details=False)

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
        cropdir = crop_save_dir+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'

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

