import os
import json, time, gc

from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import torch

MIN_CONF=0.9
VERBOSE=False
HALF=True

TIMING=False

try:
    torch.cuda.mem_get_info()
    has_nvidia_driver = True
except RuntimeError:
    has_nvidia_driver = False

def empty_gpu_cache():
    if has_nvidia_driver:
        torch.cuda.empty_cache()

def get_gpu_memory():
    if has_nvidia_driver:
        return torch.cuda.mem_get_info()
    return None, None

class Model(BaseModel):
    name: str
    version: str
    path: str

MODELS = [
    Model(name="nl_roadsigns", path="./models/nl_roadsigns.pt", version="26.0.0"),
    Model(name="fr_roadsigns", path="./models/fr_roadsigns.pt", version="0.13.0"),
]

if "MODEL_NAME" in os.environ:
    model_name = os.environ["MODEL_NAME"]
else:
    # auto detect the right model
    vram_avail, vram_total = get_gpu_memory()
    if not vram_total:
        model_name = "nl_roadsigns" # we're (testing) on CPU, use the "nano" model
    elif vram_avail < 6*(2**30):
        model_name = "nl_roadsigns"
    else:
        model_name = "nl_roadsigns"

model_config = next((m for m in MODELS if m.name == model_name), None)
if not model_config:
    raise Exception(f"Model '{model_name}' is not supported (valid models are {', '.join(m.name for m in MODELS)})")

model = YOLO(model_config.path)
print(f"loading {model_config.name} model with MIN_CONF={MIN_CONF}")

def timing(msg=''):
    if TIMING:
        print('detect:', round(time.time()-start,3), msg)

def vram_free():
    empty_gpu_cache()
    gc.collect()

def classifier(picture, cls=''):
    """Classify road signs.

    Parameters
    ----------
    picture : tempfile
		Picture file

    Returns
    -------
    Bytes
        detection result as dict
    """

    global start
    start = time.time()
    timing('classify start')
    pid = os.getpid()

    # copy received JPEG picture to temporary file
    tmp = '/dev/shm/classify%s.jpg' % pid
    result = []
    split = 0
    offset = []

    with open(tmp, 'w+b') as jpg:
        jpg.write(picture.read())

    img = Image.open(tmp)
    results = model.predict(img, conf=MIN_CONF, half=HALF, verbose=VERBOSE)
    conf = round(float(results[0].probs.top1conf),3)
    sign = results[0].names[results[0].probs.top1]

    timing('classify finished')
    print('%s (%s) found in %ss' % (sign, conf, round(time.time()-start,3)))
    return {
        'model': {'name': model_config.name,
        'version': model_config.version},
        'cls': sign,
        'conf': conf
    }
