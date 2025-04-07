import io
import json
import os
from fastapi.testclient import TestClient
import tempfile
from PIL import Image

from src.blur.blur_api import app, get_config
from src.blur.config import Config
from .conftest import FIXTURE_DIR

client = TestClient(app)


def get_overrided_config():
    tmp = tempfile.mkdtemp(prefix="test_sgblur")
    os.mkdir(f"{tmp}/crops")
    os.mkdir(f"{tmp}/tmp")
    return Config(crop_save_dir=f"{tmp}/crops", tmp_dir=f"{tmp}/tmp", detect_url="")


app.dependency_overrides[get_config] = get_overrided_config

client = TestClient(app)


def test_blur_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "GeoVisio 'Speedy Gonzales' Blurring API"}


def test_blur_picture():
    pic = FIXTURE_DIR / "flat_urban_with_face_and_sign.jpg"
    response = client.post("/blur/", files={"picture": open(pic, "rb")})
    assert response.status_code == 200

    assert "x-sgblur" in response.headers
    detection = json.loads(response.headers["x-sgblur"])
    assert "salt" in detection
    assert (
        len(detection["info"]) > 0
    )  # not much assert here, we do not test the detection model, we just check that the API returns something

    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0

    # and the returned picture is a valid JPEG
    p = Image.open(io.BytesIO(response.content))
    assert p.format == "JPEG"
