# Blur algorithms

GeoVisio blurring module offers various blur strategies. It can be selected using `STRATEGY` environment variable, which accepts `FAST`, `COMPROMISE`, `QUALITATIVE` and `LEGACY` values.

If you use __Docker__, you may want to map container folder `/opt/blur/models` to an host folder where there is enough space to store blurring models.

## Options

Several specific options are available, and can be defined using environment variables.

- `STRATEGY` (optional) : set algorithm to use for blurring, it depends on the level of speed and precision you require. Values are:
  - `DISABLE` completely disables blurring
  - `FAST` is the fastest algorithm but should be considered unreliable on pictures with lots of persons or large pictures like 360° images with persons in the background
  - `COMPROMISE` should be reliable enough for all kinds of images but the blur may be jagged
  - `QUALITATIVE` takes a lot more time to complete but will accomplish good detouring
  - `LEGACY` theorically has the best results but is *way* slower than every other method (this is the blur algorithm used in GeoVisio versions <= 1.2.0)
- `WEBP_METHOD` : quality/speed trade-off for WebP encoding of pictures derivates (0=fast, 6=slower-better, 6 by default).

## Development notes

This part presents more information about blur strategies implementation, this is a recommended-but-not-mandatory read 😉

### General blur pipeline

The blurring pipeline is as follow:

1. Get an input image
2. Split it into multiple quads, which sizes correspond to the object detection AI model (inferer)'s input size
3. Give the quads to the inferer
4. Get boxes that *may* contain cars, persons...
5. Place them on a new single image, which size is as close as possible to the sementic segmentation AI model (segmenter)'s input size
6. Give that image to the segmenter
7. Get a blur mask back (containing a blur mask for each box)
8. Arrange the blur masks back on the original image, given the inferer's boxes
9. Apply the blur mask to the original image

The inferer is [YOLOv6](https://github.com/meituan/YOLOv6) and the segmenter depends on the segmentation strategy.

When using the `FAST` strategy, steps 2-5 and 8 are skipped and inferer is not used.

On step 6 (no matter if steps 2-5 took place), if the image is larger than the segmenter's input size, it is reduced to fit in width *or* height and fed to the segmenter in one or more batches.

### Logs

There are several log messages comming from Tensorflow, Pytorch and YOLO that cannot be easily suppressed :\
`INFO: Created TensorFlow Lite XNNPACK delegate for CPU.`, see [a relating issue](https://github.com/google/mediapipe/issues/2354). To fix, a Tensorflow build from source is required.\
`...UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument...`, this is a YOLOv6 generated warning, see [a fix](https://github.com/pytorch/pytorch/issues/50276). To fix, add `, indexing='ij'` in the calls to `torch.meshgrid()` in files `effidehead.py` and `loss.py` of YOLOv6's source.

### GPU/TPU usage

To run the segmenter on the GPU, a CUDA capable GPU is required, run `pip uninstall tensorflow && pip install tensorflow-gpu`. For more information see [the Tensorflow GPU guide](https://www.tensorflow.org/guide/gpu).

To run the inferer on the GPU, refer to the [Pytorch download page](https://pytorch.org/) to find the right Pytorch version with CUDA capabilities.
Both the inferer and segmenter should automatically switch to GPU, this __hasn't been tested__ in a production environment. If you see any issues, please [let us know](https://gitlab.com/PanierAvide/geovisio/-/issues).

For TPU usage, refer to the [Tensorflow tpu guide](https://www.tensorflow.org/guide/tpu), Pytorch seems to be compatible with TPUs but it remains to be verified whether YOLO's architectures also is.

### On YOLOv6 updates

Current GeoVisio code and documentation fixes YOLOv6 to its 0.2.0 release. This has been done because trained models are not compatible through versions, and should be explicitly changed in some parts of the code. When updating, make sure to update models download links and YOLOv6 release tag everywhere necessary in GeoVisio code.
