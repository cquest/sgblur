FROM python:3.10-slim

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

COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

# Source files
COPY ./src ./src
COPY ./scripts ./scripts
COPY ./models ./models
COPY ./demo.html ./
COPY ./docker/docker-entrypoint.sh ./
RUN chmod +x ./docker-entrypoint.sh

# Expose service
EXPOSE 8001
ENTRYPOINT ["./docker-entrypoint.sh"]
