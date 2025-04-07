from functools import lru_cache
from typing import Annotated, Optional
from fastapi import Depends, FastAPI, Header, UploadFile, Response, HTTPException
import urllib3
import gc
import json

from . import blur, semantics
from .config import Config

app = FastAPI()
print("API is preparing to start...")


@app.get("/")
async def root():
	return {"message": "GeoVisio 'Speedy Gonzales' Blurring API"}

@lru_cache
def get_config():
    return Config()

@app.post(
	"/blur/",
	responses = {200: {"content": {
		"image/jpeg": {},
		"multipart/form-data": {},
		}}},
	response_class=Response
)
<<<<<<< HEAD
async def blur_picture(picture: UploadFile, debug: str|None='0' , keep: str|None='0'):
	blurredPic, blurInfo = blur.blurPicture(picture, keep, debug)
=======
async def blur_picture(picture: UploadFile, config: Annotated[Config, Depends(get_config)], keep: str | None = '0', accept: Optional[str] = Header("image/png")): 
	"""Blur a picture.
	The picture must be a JPEG file.

	If the `accept` header is set to `multipart/form-data`, the returned picture will be a multipart/form-data, containing the blurred picture and some semantic tags detailling what has been detected in the picture.
	Otherwise, the returned picture will be a JPEG file and the detection are send in the `x-sgblur` header.
	"""
	blurredPic, blurInfo = blur.blurPicture(picture, keep, config=config)
>>>>>>> 35264b1 (Change detections to semantic tags)

	# For some reason garbage collection does not run automatically after
	# a call to an AI model, so it must be done explicitely
	gc.collect()

	if not blurredPic:
		raise HTTPException(status_code=400, detail=(blurInfo if blurInfo else "Invalid picture to process"))

	if "multipart/form-data" in accept:
		detection_info = semantics.detection_to_tags(blurInfo, config)
		content, content_type = urllib3.encode_multipart_formdata(fields={
			'detections':("detections", json.dumps(detection_info), {"Content-Type": "application/json"}),
			'image': ('filename', blurredPic, 'image/jpeg'),
		})

		return Response(content=content, media_type=content_type)

	headers = { "x-sgblur": json.dumps(blurInfo)}
	return Response(content=blurredPic, media_type="image/jpeg", headers=headers)


@app.get(
	"/blur/",
	responses={200: {"content": {"text/html": {}}}},
	response_class=Response
)
async def blur_form():
	return Response(content=open('demo.html', 'rb').read(), media_type="text/html")


@app.head(
	"/blur/",
	responses={200: {"content": {"text/html": {}}}},
	response_class=Response
)
async def blur_form():
	return


@app.post(
	"/deblur/",
	responses={200: {"content": {"image/jpeg": {}}}},
	response_class=Response,
)
async def deblur_picture(picture: UploadFile, idx: int, config: Annotated[Config, Depends(get_config)], salt=''):
	deblurredPic = blur.deblurPicture(picture, idx, salt, config)

	# For some reason garbage collection does not run automatically after
	# a call to an AI model, so it must be done explicitely
	gc.collect()

	if not deblurredPic:
		raise HTTPException(status_code=400, detail="Invalid picture to process")

	return Response(content=deblurredPic, media_type="image/jpeg")
