from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, Form
from fastapi.responses import JSONResponse
from tensorflow import keras
import cv2
import numpy as np
import requests
from io import BytesIO

app = FastAPI()

model = keras.models.load_model('/Users/zacharycassidy/Downloads/fruit_ripeness_model (2).h5')

def download_image_from_url(url: str) -> np.ndarray:
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image from URL: {str(e)}")

@app.post("/classify")
async def classify_fruit(url: str = Form(...)):
    # Download the image from the URL
    image = download_image_from_url(url)

    # Preprocess the image
    img = cv2.resize(image, (224, 224))  # Adjust the size as per your model's input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(img)

    # Assuming binary classification (fresh or rotten)
    result = "Fresh" if prediction[0][0] < 0.5 else "Rotten"

    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



