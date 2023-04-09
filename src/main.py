from pathlib import Path
from enum import Enum
import typer
from PIL import Image
from blur import blur
from contextlib import contextmanager
from rich import print


class Strategy(str, Enum):
    fast = "fast"
    legacy = "legacy"
    compromise = "compromise"
    qualitative = "qualitative"


app = typer.Typer(help="GeoVisio blurring scripts")


@contextmanager
def log_elapsed(ctx: str):
    """Context manager used to log the elapsed time of the context

    Args:
        ctx (str): Label to describe what is timed
    """
    from time import perf_counter
    from datetime import timedelta
    start = perf_counter()
    yield
    print(f"⏲ [bold]{ctx}[/bold] done in {timedelta(seconds=perf_counter()-start)}")


@app.callback()
def main(
    input: Path = typer.Argument(..., help="Picture to blur"),
    output: Path = typer.Argument(..., help="Output file path"),
    strategy: Strategy = typer.Option(Strategy.fast, help="Blur algorithm to use"),
    mask: bool = typer.Option(False, "--mask/--picture", help="Get a blur mask instead of blurred picture"),
) -> None:
    """Creates a blurred version of a picture"""

    config = {"BLUR_STRATEGY": strategy.upper(), "MODELS_FS_URL": "./models"}

    with log_elapsed("Model initialization"):
        blur.blurPreinit(config)

    picture = Image.open(input)
    with log_elapsed("Blur mask generation"):
        blurMask = blur.getBlurMask(picture)

    if mask:
        img = blurMask
    else:
        with log_elapsed("Application of mask on picture"):
            img = blur.blurPicture(picture, blurMask)

    with log_elapsed("Blurred picture saving"):
        img.save(output, format="png", optimize=True, bits=1)


if __name__ == "__main__":
    typer.run(main)
