# `python src/main.py`

GeoVisio blurring scripts

**Usage**:

```console
$ python src/main.py [OPTIONS] INPUT OUTPUT COMMAND [ARGS]...
```

**Arguments**:

* `INPUT`: Picture to blur  [required]
* `OUTPUT`: Output file path  [required]

**Options**:

* `--strategy [fast|legacy|compromise|qualitative]`: Blur algorithm to use  [default: Strategy.fast]
* `--mask / --picture`: Get a blur mask instead of blurred picture  [default: picture]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.
