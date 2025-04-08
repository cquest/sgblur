FROM python:3.12-slim

WORKDIR /opt/blur

# Dependencies
RUN apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    # * Pillow
    libffi-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-turbo-progs \
    libjpeg62-turbo-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libwebp-dev \
    libssl-dev \
    meson \
    netpbm \
    xvfb \
    zlib1g-dev \
    # * sgblur
    libturbojpeg0-dev \
    libjpeg-turbo-progs \
    exiftran \
    && rm -rf /var/lib/apt/lists/*

COPY ./pyproject.toml ./
RUN pip install -e .

# Source files
COPY ./src ./src
COPY ./scripts ./scripts
COPY ./models ./models
COPY ./demo.html ./

ENV CROP_SAVE_DIR=/data/crops
ENV TMP_DIR=/dev/shm

RUN mkdir -p $CROP_SAVE_DIR $TMP_DIR

# Expose service
EXPOSE 8000
 
ENTRYPOINT ["uvicorn", "src.blur.blur_api:app", "--host", "0.0.0.0", "--port", "8000"]
