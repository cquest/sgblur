from fastapi import FastAPI, UploadFile, HTTPException
import gc
from . import classify

app = FastAPI()
print("API is preparing to start...")

@app.get("/")
async def root():
	return {"message": "GeoVisio 'Speedy Gonzales' Road signs classification API"}

@app.post("/classify/")
async def classify_api(picture:UploadFile, cls:str = ''):
	result = classify.classifier(picture.file, cls)

	# For some reason garbage collection does not run automatically after
	# a call to an AI model, so it must be done explicitely
	gc.collect()

	if not result:
		raise HTTPException(status_code=400, detail="Invalid picture to process")
    
	return result
