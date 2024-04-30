from fastapi import FastAPI, File, UploadFile
import pytesseract
import cv2
import numpy as np

# Initialize FastAPI application
app = FastAPI()

@app.post("/ocr/extract_text/")
async def extract_text(image: UploadFile = File(...), lang: str = None):
  """
  API endpoint to perform OCR on an uploaded image.
  """
  try:
    # Read the image from the uploaded file
    content = await image.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Text extraction
    text = pytesseract.image_to_string(gray, lang=lang)

    # Return the extracted text as a JSON response
    return {"text": text}

  except Exception as e:
    # Handle errors gracefully, like file type errors or OCR failures
    return {"error": str(e)}


@app.get("/ocr/languages/")
def get_languages():
  """
  API endpoint to get the list of supported languages for OCR.
  """

  return {
    "English": "end",
    "Russian": "rus",
    "German": "ger",
    "French": "fra",
    "Italian": "ita",
    "Turkish": "tur",
    "Uzbek": "uzb",
    "Chinese Simplified": "chi_sim",
    "Chinese Traditional": "chi_tra",
    "Arabic": "ara",
    "Spanish": "spa",
    "Portuguese": "por",
    "Japanese": "jpn",
  }


# Run the API
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
