from fastapi import FastAPI, UploadFile, Response
import gc
from .blur import blur
import json

app = FastAPI()
print("API is preparing to start...")


@app.get("/")
async def root():
	return {"message": "GeoVisio 'Speedy Gonzales' Blurring API"}


@app.post(
	"/blur/",
	responses = {200: {"content": {"image/jpeg": {}}}},
	response_class=Response
)
async def blur_picture(picture: UploadFile):
	blurredPic, blurInfo = blur.blurPicture(picture)

	# For some reason garbage collection does not run automatically after
	# a call to an AI model, so it must be done explicitely
	gc.collect()

	if not blurredPic:
		raise HTTPException(status_code=400, detail="Invalid picture to process")

	headers = { "x-blur": json.dumps(blurInfo)}
	return Response(content=blurredPic, media_type="image/jpeg", headers=headers)


@app.get(
	"/blur/",
	responses={200: {"content": {"text/html": {}}}},
	response_class=Response
)
async def blur_form():
	return Response(content=open('demo.html', 'rb').read(), media_type="text/html")
