from fastapi import FastAPI, UploadFile, File
from engine import analyze_crop_leaf, get_prediction_history

app = FastAPI(title="CropVision AI Backend API")

@app.post("/predict")
async def predict_leaf(file: UploadFile = File(...)):
    contents = await file.read()
    return analyze_crop_leaf(contents, filename=file.filename)

@app.get("/history")
def get_history():
    rows = get_prediction_history(limit=10)
    return {"history": rows}
