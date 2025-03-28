import os, subprocess
from datetime import datetime

import turbojpeg
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import hashlib, pathlib, time
import exifread
import json, uuid
import requests
import piexif, piexif.helper

DEBUG=False
TIMING=False

def timing(msg=''):
    if TIMING:
        print('blur:', round(time.time()-start,3),msg)


# JPEGTRAN_OPTS='-progressive -optimize -copy all'
JPEGTRAN_OPTS='-optimize -copy all'

jpeg = turbojpeg.TurboJPEG()

crop_save_dir = 'saved_crops'

def copytags(src, dst, comment=None):
    if False:
        comment = ' -Comment=\'%s\' ' % comment if comment else ''
        subprocess.run('-overwrite_original -tagsfromfile %s %s %s' % (src, comment, dst), shell=True)
    else:
        tags = piexif.load(src)
        tags['thumbnail'] = None
        if comment:
            tags["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(comment)
        piexif.insert(piexif.dump(tags), dst)

def blurPicture(picture, keep):
    """Blurs a single picture by detecting faces and licence plates.

    Parameters
    ----------
    picture : tempfile
		Picture file
    keep : str
        1 to keep blurred part to allow deblur
        2 keep detected road signs only, do not blur

    Returns
    -------
    Bytes
        the blurred image
    """

    pid = os.getpid()
    global start
    start = time.time()
    timing('start')

    # copy received JPEG picture to temporary file
    tmp = '/dev/shm/blur%s.jpg' % pid
    tmpcrop = '/dev/shm/crop%s.jpg' % pid

    nb_blurred = 0
    nb_saved = 0

    with open(tmp, 'w+b') as jpg:
        jpg.write(picture.file.read())
        jpg.seek(0)
        try:
            tags = exifread.process_file(jpg, details=False)
        except:
            tags = None
    if DEBUG:
        print("keep", keep, "original", os.path.getsize(tmp))

    # solve image orientation
    if tags and 'Image Orientation' in tags:
        if ('normal' not in str(tags['Image Orientation'])
                and str(tags['Image Orientation'])!='0'):
            subprocess.run('exiftran -a %s -o %s' % (tmp, tmp+'_tmp'), shell=True)
            if DEBUG:
                print("after exiftran", os.path.getsize(tmp+'_tmp'))
            os.replace(tmp+'_tmp', tmp)

    # get picture details
    with open(tmp, 'rb') as jpg:
        width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(
            jpg.read())


    # call the detection microservice
    if DEBUG:
        timing('call detection')
    try:
        files = {'picture': open(tmp,'rb')}
        if keep == '2':
            r = requests.post('http://localhost:8001/detect/?cls=sign', files=files)
        else:
            r = requests.post('http://localhost:8001/detect/', files=files)
        results = json.loads(r.text)
        info = results['info']
        crop_rects = results['crop_rects']
        bbox = results['bbox'] if 'bbox' in results else None
    except:
        return None

    salt = None

    # get MCU maximum size (2^n) = 8 or 16 pixels subsamplings
    hblock, vblock, sample = [(3, 3 ,'1x1'), (4, 3, '2x1'), (4, 4, '2x2'), (4, 4, '2x2'), (3, 4, '1x2')][jpeg_subsample]

    today = datetime.today().strftime('%Y-%m-%d')
    if DEBUG:
        timing()
        print(len(crop_rects), 'detections')
    if len(crop_rects)>0:
        # extract cropped jpeg data from boxes to be blurred
        with open(tmp, 'rb') as jpg:
            crops = jpeg.crop_multiple(jpg.read(), crop_rects, background_luminance=0, copynone=True)

        # if face or plate, blur boxes and paste them onto original
        for c in range(len(crops)):
            if keep=='2':
                break
            if DEBUG:
                print(info[c]['class'], crop_rects[c])
            if info[c]['class'] == 'sign':
                continue

            bbox = info[c]['bbox']
            xywh = info[c]['xywh']
            box = (bbox[0]-xywh[0], bbox[1]-xywh[1],bbox[2]-xywh[0], bbox[3]-xywh[1])

            if xywh[2] < 12 and xywh[3] < 12:
                if DEBUG:
                    print('too small, skip')
                continue

            nb_blurred += 1
            crop = open(tmpcrop,'wb')
            crop.write(crops[c])
            crop.close()
            # pillow based blurring
            img = Image.open(tmpcrop)
            ccrop = img.crop(box)
            radius = max(int(max(img.width, img.height)/12) >> 3 << 3, 8)
            # pixelate first
            reduced = ImageOps.scale(ccrop, 1/radius, resample=0)
            pixelated = ImageOps.scale(reduced, radius, resample=0)
            # and blur
            boxblur = pixelated.filter(ImageFilter.BoxBlur(radius))
            img.paste(boxblur, (bbox[0]-xywh[0], bbox[1]-xywh[1]))
            if DEBUG:
                draw = ImageDraw.Draw(img)
                draw.rectangle(box, outline=(0, 255, 255))
            img.save(tmpcrop, subsampling=jpeg_subsample)
            subprocess.run('djpeg %s | cjpeg -sample %s -optimize -dct float -baseline -outfile %s' % (tmpcrop, sample, tmpcrop+'_tmp'), shell=True)
            os.replace(tmpcrop+'_tmp', tmpcrop)

            # jpegtran "drop"
            if DEBUG:
                timing()
                print( 'crop size', os.path.getsize(tmpcrop))
            p = subprocess.run('jpegtran %s -trim -drop +%s+%s %s %s > %s' % (JPEGTRAN_OPTS, crop_rects[c][0], crop_rects[c][1], tmpcrop, tmp, tmp+'_tmp'), shell=True)
            if DEBUG:
                timing()
                print( 'after drop', os.path.getsize(tmp+'_tmp'))
            if p.returncode != 0 :
                if DEBUG:
                    print('crop %sx%s -> recrop %sx%s' % (img.width, img.height, crop_rects[c][2], crop_rects[c][3]))
                subprocess.run('jpegtran -crop %sx%s+0+0 %s > %s' % (img.width, img.height, tmpcrop, tmpcrop+'_tmp'), shell=True)
                p = subprocess.run('jpegtran %s -trim -drop +%s+%s %s %s > %s' % (JPEGTRAN_OPTS, crop_rects[c][0], crop_rects[c][1], tmpcrop+'_tmp', tmp, tmp+'_tmp'), shell=True)
                if img.height != crop_rects[c][3]:
                    input()

            if p.returncode != 0 :
                if DEBUG:
                    timing()
                    print('>>>>>> crop info: ',info[c])
                # problem with original JPEG... we try to recompress it
                subprocess.run('djpeg %s | cjpeg -optimize -smooth 10 -dct float -baseline -outfile %s' % (tmp, tmp+'_tmp'), shell=True)
                # copy EXIF tags
                copytags(tmp, tmp+'_tmp')
                if DEBUG:
                    timing()
                    print('after recompressing original', os.path.getsize(tmp+'_tmp'))
                os.replace(tmp+'_tmp', tmp)
                if DEBUG:
                    timing()
                    print('jpegtran %s -trim -drop +%s+%s %s %s > %s' % (JPEGTRAN_OPTS, crop_rects[c][0], crop_rects[c][1], tmpcrop, tmp, tmp+'_tmp'))
                subprocess.run('jpegtran %s -trim -drop +%s+%s %s %s > %s' % (JPEGTRAN_OPTS, crop_rects[c][0], crop_rects[c][1], tmpcrop, tmp, tmp+'_tmp'), shell=True)
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
                    nb_saved += 1
                    h = hashlib.sha256()
                    h.update(((salt if not info[c]['class'] == 'sign' else str(info))+str(info[c])).encode())
                    cropname = h.hexdigest()+'.jpg'
                    if info[c]['class'] != 'sign':
                        dirname = crop_save_dir+'/'+info[c]['class']+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'
                    else:
                        dirname = crop_save_dir+'/'+info[c]['class']+'/'+today+'/'+str(round(info[c]['confidence'],1))+'/'
                    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
                    with open(dirname+cropname, 'wb') as crop:
                        crop.write(crops[c])
                    if info[c]['class'] == 'sign':
                        if DEBUG:
                            timing()
                            print('copy EXIF')
                        info2 = {   'width':width,
                                    'height':height,
                                    'xywh': info[c]['xywh'],
                                    'confidence': info[c]['confidence'],
                                    'offset': round((info[c]['xywh'][0]+info[c]['xywh'][2]/2.0)/width - 0.5,3)}
                        # subprocess.run('exiv2 -ea %s | exiv2 -ia %s' % (tmp, dirname+cropname), shell=True)
                        comment = json.dumps(info2, separators=(',', ':'))
                        if DEBUG:
                            print(dirname+cropname, comment)
                        copytags(tmp, dirname+cropname, comment=comment)
                    else:
                        # round ctime/mtime to midnight
                        daytime = int(time.time()) - int(time.time()) % 86400
                        os.utime(dirname+cropname, (daytime, daytime))
            info = { 'info': info, 'salt': salt }

        if False:
            # regenerate EXIF thumbnail
            before_thumb = os.path.getsize(tmp)
            subprocess.run('exiftran -g -o %s %s' % (tmp+'_tmp', tmp), shell=True)
            os.replace(tmp+'_tmp', tmp)
            if DEBUG:
                timing()
                print("after thumbnail", os.path.getsize(tmp), (100*(os.path.getsize(tmp)-before_thumb)/before_thumb))

    # return result (original image if no blur needed)
    with open(tmp, 'rb') as jpg:
        original = jpg.read()

    if not DEBUG:
        try:
            os.remove(tmp)
            os.remove(tmpcrop)
        except:
            pass

    timing('end')
    # summary output
    print('%s Mpx picture, %s blur, %s saved in %ss' % (round(width/1024.0*height/1024.0,1), nb_blurred, nb_saved, round(time.time()-start,3)))
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

