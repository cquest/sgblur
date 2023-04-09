# GeoVisio Blurring Algorithms

![GeoVisio logo](https://gitlab.com/geovisio/api/-/blob/52df05fd9c95f1a929b32bee9735505eeeddc7e8/images/logo_full.png)

[GeoVisio](https://gitlab.com/geovisio) is a complete solution for storing and __serving your own geolocated pictures__ (like [StreetView](https://www.google.com/streetview/) / [Mapillary](https://mapillary.com/)).

This repository only contains __the blurring algorithms and API__. This repository can be used completely independently from others. [All other components are listed here](https://gitlab.com/geovisio).


## Install

### System dependencies

Some algorithms (compromise and qualitative) will need system dependencies :

- ffmpeg
- libsm6
- libxext6

You can install them through your package manager, for example in Ubuntu:

```bash
sudo apt install ffmpeg libsm6 libxext6
```

### Retrieve code

You can download code from this repository with git clone:

```bash
git clone https://gitlab.com/geovisio/blurring.git
cd blurring/
```

### Other dependencies

We use [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) to manage some of our dependencies. Run the following command to get these dependencies:

```bash
git submodule update --init
```

We also use Pip to handle Python dependencies. You can create a virtual environment first:

```bash
python -m venv env
source ./env/bin/activate
```

And depending on if you want to use API, command-line scripts or both, run these commands:

```bash
pip install -r requirements-bin.txt  # For CLI
pip install -r requirements-api.txt  # For API
```

If at some point you're lost or need help, you can contact us through [issues](https://gitlab.com/geovisio/blurring/-/issues) or by [email](mailto:panieravide@riseup.net).


## Usage

### Command-line interface

All details of available commands are listed in [USAGE.md](./USAGE.md) documentation, or by calling this command:

```bash
python src/main.py --help
```

A single picture can be blurred using the following command:

```bash
python src/main.py <path the the picture> <path to the output picture>
```

You can also launch the CLI through Docker:

```bash
docker run \
	geovisio/blurring \
	cli
```

### Web API

The Web API can be launched with the following command:

```bash
uvicorn src.api:app --reload
```

It is then accessible on [localhost:8000](http://127.0.0.1:8000).

You can also launch the API through Docker:

```bash
docker run \
	-p 8000:80 \
	--name geovisio_blurring \
	geovisio/blurring \
	api
```

API documentation is available under `/docs` route, so [localhost:8000/docs](http://127.0.0.1:8000/docs) if you use local instance.

A single picture can be blurred using the following HTTP call (here made using _curl_):

```bash
# Considering your picture is called my_picture.jpg
curl -X 'POST' \
  'http://127.0.0.1:8000/blur/' \
  -H 'accept: image/webp' \
  -H 'Content-Type: multipart/form-data' \
  -F 'picture=@my_picture.jpg;type=image/jpeg' \
  --output blurred.webp
```

Note that various settings can be changed to control API behaviour. You can edit them [using one of the described method in FastAPI documentation](https://fastapi.tiangolo.com/advanced/settings/). Available settings are:

- `STRATEGY`: blur algorithm to use (FAST, LEGACY, COMPROMISE, QUALITATIVE)
- `WEBP_METHOD`: quality/speed trade-off for WebP encoding of pictures derivates (0=fast, 6=slower-better, 6 by default)


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

You might want to read more [about available blur algorithms](./ALGORITHMS.md).

### Testing

Tests are handled with Pytest. You can run them using:

```bash
pip install -r requirements-dev.txt
pytest
```

### Documentation

High-level documentation for command-line script is handled by [Typer](https://typer.tiangolo.com/). You can update the generated `USAGE.md` file using this command:

```bash
make docs
```


## License

Copyright (c) GeoVisio team 2022-2023, [released under MIT license](./LICENSE).
