# Panoramax "Speedy Gonzales" Blurring Algorithm

This repository only contains __the blurring algorithms and API__.


## Install

### System dependencies

Some algorithms (compromise and qualitative) will need system dependencies :

- ffmpeg
- libsm6
- libxext6

You can install them through your package manager, for example in Ubuntu:

```bash
sudo apt install git libturbojpeg0-dev libjpeg-turbo-progs python-is-python3 python3-pip
```

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

Intall python depedencies for the API:

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

http --form POST http://127.0.0.1:8000/blur/ picture@original.jpg --download --oupput blurred.jpg



## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

You might want to read more [about available blur algorithms](./ALGORITHMS.md).


## License

Copyright (c) GeoVisio/panoramax team 2022-2023, [released under MIT license](./LICENSE).
