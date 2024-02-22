# Panoramax "Speedy Gonzales" Blurring Algorithm

This repository only contains __the blurring algorithms and API__.

It is based on YOLOv8 for object detection (faces and license plates) using a custom trained model.

Blurring is done on original JPEG pictures by manipulating low-level MCU in the JPEG raw data, to keep all other parts of the original image unchanged (no decompression/recompression). This also saves CPU usage.


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
pip install -r requirements-api.txt
```



## Usage

### Web API

The Web API can be launched with the following command:

```bash
uvicorn src.api:app --reload
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

Exemple using httpie :

```bash
http --form POST http://127.0.0.1:8000/blur/ picture@original.jpg --download --output blurred.jpg
```

A **demo API** is running on https://api.cquest.org/blur/


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

You might want to read more [about available blur algorithms](./ALGORITHMS.md).


## License

Copyright (c) GeoVisio/panoramax team 2022-2023, [released under MIT license](./LICENSE).
