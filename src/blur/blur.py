import os, subprocess
from datetime import datetime
import logging
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

from .config import Config

jpeg = turbojpeg.TurboJPEG()

def copytags(src, dst, comment=None):
    tags = piexif.load(src)
    tags['thumbnail'] = None
    if comment:
        tags["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(comment)
    try:
        piexif.insert(piexif.dump(tags), dst)
    except:
        print('>> copytags retry')
        # when tag copy fails, only copy the minimum we need
        exif_ifd = {
            piexif.ExifIFD.DateTimeOriginal: tags['Exif'][piexif.ExifIFD.DateTimeOriginal],
            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(comment)
        }
        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSLatitude: tags['GPS'][piexif.GPSIFD.GPSLatitude],
            piexif.GPSIFD.GPSLatitudeRef: tags['GPS'][piexif.GPSIFD.GPSLatitudeRef],
            piexif.GPSIFD.GPSLongitude: tags['GPS'][piexif.GPSIFD.GPSLongitude],
            piexif.GPSIFD.GPSLongitudeRef: tags['GPS'][piexif.GPSIFD.GPSLongitudeRef],
            piexif.GPSIFD.GPSDateStamp: (tags['GPS'][piexif.GPSIFD.GPSDateStamp]
                if piexif.GPSIFD.GPSDateStamp in tags['GPS']
                else tags['Exif'][piexif.ExifIFD.DateTimeOriginal]),
        }
        piexif.insert(piexif.dump({"Exif": exif_ifd, "GPS":gps_ifd, "thumbnail":None}), dst)

def detect(picture, keep, config: Config):
    params = {'cls': "sign"} if keep == '2' else {}
    if config.detect_url != '':
        # call the detection microservice
        files = {'picture': open(picture,'rb')}
        r = requests.post(f'{config.detect_url}/detect/', files=files, params=params)
        r.raise_for_status()
        return r.json()
    else:
        from src.detect import detect
        # call directly the detection code
        with open(picture, 'rb') as f:
            return detect.detector(f, **params)
    
def blurPicture(picture, keep, debug, config: Config = Config()):
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
    DEBUG = (debug != '0')
    # copy received JPEG picture to temporary file
    tmp = config.tmp_dir + '/blur%s.jpg' % pid
    tmpcrop = config.tmp_dir + '/crop%s.jpg' % pid

    nb_blurred = 0
    nb_saved = 0

    with open(tmp, 'w+b') as jpg:
        jpg.write(picture.file.read())

        # check for premature end of JPEG
        jpg.seek(0, os.SEEK_END)
        jpg.seek(-2, os.SEEK_CUR)
        trailer = jpg.read(2)
        jpg.seek(0)
        try:
            tags = exifread.process_file(jpg, details=False)
        except:
            return None,"Can't read EXIF tags (exifread failed)"

    if trailer != b'\xFF\xD9':
        try:
            # call jpegoptim to cleanup the JPEG file (and lossless optimize it)
            subprocess.run('jpegoptim --strip-none %s ' % tmp, shell=True)
        except:
            print('premature end of JPEG data')
            return None,'premature end of JPEG data, missing 0xFFD9 at end of file'

    with open(tmp, 'rb') as jpg:
        jpg.seek(0, os.SEEK_END)
        jpg.seek(-2, os.SEEK_CUR)
        if jpg.read(2) != b'\xFF\xD9':
                print('premature end of JPEG data after jpegoptim')
                return None,'premature end of JPEG data, missing 0xFFD9 at end of file'

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
    try:
        with open(tmp, 'rb') as jpg:
            width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(
                jpg.read())
    except:
        return None,"Can't decode JPEG header, corrupted file ?"

    # call the detection microservice
    if DEBUG:
        timing('call detection')
    try:
        results = detect(tmp, keep, config)
    except Exception as e:
        logging.error(f"Impossible to detect picture, error = {e}")
        return None, 'detection failed'
    info = results['info']
    crop_rects = results.pop('crop_rects')

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

            if bbox[2]-bbox[0] < 12 or bbox[3]-bbox[1] < 12:
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
                draw.rectangle(box, outline=(255, 0, 255), width=3)
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
                    print('sampling', sample)
                    print('crop %sx%s -> recrop %sx%s' % (img.width, img.height, crop_rects[c][2], crop_rects[c][3]))
                subprocess.run('jpegtran -crop %sx%s+0+0 %s > %s' % (img.width, img.height, tmpcrop, tmpcrop+'_tmp'), shell=True)
                p = subprocess.run('jpegtran %s -trim -drop +%s+%s %s %s > %s' % (JPEGTRAN_OPTS, crop_rects[c][0], crop_rects[c][1], tmpcrop+'_tmp', tmp, tmp+'_tmp'), shell=True)
                if p.returncode != 0 :
                    print('recompress original')
                    subprocess.run('djpeg %s | cjpeg -sample %s -quality 80 -optimize -dct float -baseline -outfile %s' % (tmp, sample, tmp+'_tmp'), shell=True)
                    copytags(tmp, tmp+'_tmp')
                    os.replace(tmp+'_tmp', tmp)
                    p = subprocess.run('jpegtran %s -trim -drop +%s+%s %s %s > %s' % (JPEGTRAN_OPTS, crop_rects[c][0], crop_rects[c][1], tmpcrop+'_tmp', tmp, tmp+'_tmp'), shell=True)
                    if p.returncode != 0 :
                        return None,'JPEG recompression failed'

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
        if config.crop_save_dir != '':
            salt = str(uuid.uuid4())
            for c in range(len(crops)):
                if ((keep == '1' and info[c]['confidence'] < 0.5 and info[c]['class'] in ['face', 'plate'])
                        or (info[c]['confidence'] > 0.2 and info[c]['class'] == 'sign')):
                    nb_saved += 1
                    h = hashlib.sha256()
                    h.update(((salt if not info[c]['class'] == 'sign' else str(info))+str(info[c])).encode())
                    cropname = h.hexdigest()+'.jpg'
                    if info[c]['class'] != 'sign':
                        dirname = config.crop_save_dir+'/'+info[c]['class']+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'
                    else:
                        dirname = config.crop_save_dir+'/'+info[c]['class']+'/'+today+'/'+str(round(info[c]['confidence'],1))+'/'
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
                        try:
                            copytags(tmp, dirname+cropname, comment=comment)
                        except:
                            os.remove(dirname+cropname)
                            print('copytags failed')
                    else:
                        # round ctime/mtime to midnight
                        daytime = int(time.time()) - int(time.time()) % 86400
                        os.utime(dirname+cropname, (daytime, daytime))
            results['salt'] = salt 

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
    print('%s Mpx picture, %s blur, %s saved in %ss' % (round(width*height/1000000.0,1), nb_blurred, nb_saved, round(time.time()-start,3)))
    return original, results


def deblurPicture(picture, idx, salt, config):
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
        tmp = config.tmp_dir + '/deblur%s.jpg' % pid

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
        cropdir = config.crop_save_dir+'/'+i[idx]['class']+'/'+cropname[0:2]+'/'+cropname[0:4]+'/'

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

