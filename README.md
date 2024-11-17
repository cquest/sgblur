# Panoramax "Speedy Gonzales" Blurring Algorithm

This repository only contains __the blurring algorithms and API__.

It is based on YOLOv8 for object detection (faces and license plates) using a custom trained model.

Blurring is done on original JPEG pictures by manipulating low-level MCU in the JPEG raw data, to keep all other parts of the original image unchanged (no decompression/recompression). This also saves CPU usage.

The blurring service call a detection service with through local HTTP calls.


## Install

### System dependencies

These dependencies are needed for lossless JPEG transformations :

- turbojpeg library and headers
- exiftran

You can install them through your package manager, for example in Ubuntu:

```bash
sudo apt install libturbojpeg0-dev libjpeg-turbo-progs exiftran
```

Basic dependencies may also need:

```bash
sudo apt install git python-is-python3 python3-pip
```

Running on a GPU will requires NVidia drivers and Cuda.


### Retrieve code

You can download code from this repository with git clone:

```bash
git clone https://github.com/cquest/sgblur.git
cd sgblur/
```

### Other dependencies

We use Pip to handle Python dependencies. You can create a virtual environment first:

```bash
python -m venv env
source ./env/bin/activate
```

Install python dependencies for the API:

```bash
pip install -r requirements.txt
```

## Usage

### Web API

The Web API can be launched with the following commands:

```bash
# detection service on port 8001 (1 worker to save GPU VRAM)
uvicorn src.detect.detect_api:app --port 8001

# blurring service (several workers using CPU for the blurring)
uvicorn src.blur.blur_api:app --workers 8 
```

It is then accessible on [localhost:8000](http://127.0.0.1:8000).

A single picture can be blurred using the following HTTP call (here made using _curl_):

```bash
# Considering your picture is called original.jpg
curl -X 'POST' \
  'http://127.0.0.1:8000/blur/' \
  -F 'picture=@original.jpg' \
  --output blurred.jpg
```

Example using httpie :

```bash
http --form POST http://127.0.0.1:8000/blur/ picture@original.jpg --download --output blurred.jpg
```

A **demo API** with a minimal web UI is running on https://panoramax.openstreetmap.org/blur/

**DO NOT USE** it in production without prior authorization. Thanks.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

You might want to read more [about available blur algorithms](./ALGORITHMS.md).


## License

Copyright (c) Panoramax team 2022-2024, [released under MIT license](./LICENSE).
