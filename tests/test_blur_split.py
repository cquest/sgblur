import io
import json
import os
from fastapi.testclient import TestClient
import tempfile
from PIL import Image
import multipart
import pytest


from src.blur.blur_api import app, get_config
from src.blur.config import Config
from .conftest import FIXTURE_DIR

MOCK_DETECT_URI = "https://detect.panoramax.com"

client = TestClient(app)


def get_overrided_config_split():
    tmp = tempfile.mkdtemp(prefix="test_sgblur")
    os.mkdir(f"{tmp}/crops")
    os.mkdir(f"{tmp}/tmp")
    return Config(crop_save_dir=f"{tmp}/crops", tmp_dir=f"{tmp}/tmp", detect_url=MOCK_DETECT_URI)


@pytest.fixture(autouse=True, scope="module")
def override_config():
    app.dependency_overrides[get_config] = get_overrided_config_split


MOCK_DETECTION = {
    "info": [{"class": "sign", "confidence": 0.313, "xywh": [144, 96, 64, 64]}],
    "crop_rects": [[144, 96, 64, 64]],
    "model": {"name": "yolo11n", "version": "0.1.0"},
}

PANORAMAX_DETECTIONS_SEMANTICS = {
    "annotations": [
        {
            "shape": [144, 96, 64, 64],
            "semantics": [
                {"key": "osm|traffic_sign", "value": "yes"},
                {"key": "detection_model[osm|traffic_sign=yes]", "value": "SGBlur-yolo11n/0.1.0"},
                {"key": "detection_confidence[osm|traffic_sign=yes]", "value": "0.313"},
            ],
        }
    ],
    "service_name": "SGBlur",
}


def test_blur_picture(requests_mock):
    requests_mock.post(f"{MOCK_DETECT_URI}/detect/", json=MOCK_DETECTION)
    pic = FIXTURE_DIR / "flat_urban_with_face_and_sign_small.jpg"
    response = client.post("/blur/", files={"picture": open(pic, "rb")})
    assert response.status_code == 200

    assert "x-sgblur" in response.headers
    detection = json.loads(response.headers["x-sgblur"])
    assert detection["info"] == MOCK_DETECTION["info"]
    assert "salt" in detection

    assert response.headers["content-type"] == "image/jpeg"
    assert response.content

    # and the returned picture is a valid JPEG
    p = Image.open(io.BytesIO(response.content))
    assert p.format == "JPEG"


def test_blur_picture_multipart(requests_mock):
    requests_mock.post(f"{MOCK_DETECT_URI}/detect/", json=MOCK_DETECTION)
    pic = FIXTURE_DIR / "flat_urban_with_face_and_sign_small.jpg"
    response = client.post("/blur/", files={"picture": open(pic, "rb")}, headers={"Accept": "multipart/form-data"})
    assert response.status_code == 200

    # the detections are not longer in the headers, but in the body
    assert "x-sgblur" not in response.headers

    assert "multipart/form-data" in response.headers["content-type"]
    assert response.content

    content_type, boundary = multipart.parse_options_header(response.headers["content-type"])
    assert content_type == "multipart/form-data"
    multipart_response = multipart.MultipartParser(io.BytesIO(response.content), boundary=boundary["boundary"])

    metadata = multipart_response.get("metadata")
    pic = multipart_response.get("image")

    metadata = json.loads(metadata.raw)
    assert metadata["annotations"] == PANORAMAX_DETECTIONS_SEMANTICS["annotations"]
    assert metadata.get("blurring_id")  # we should also have a blurring id in the response
    assert metadata["service_name"] == "SGBlur"

    # and the returned picture is a valid JPEG
    p = Image.open(io.BytesIO(pic.raw), formats=["jpeg"])
    assert p.format == "JPEG"
