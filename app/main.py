import io       
import torch
from pydantic import BaseModel # adds type hints and checking to our data
from fastapi import FastAPI, File, UploadFile, Depends
from app.model import load_model, load_transform
from torchvision.models import ResNet
from torchvision.transforms import v2 as transforms
from PIL import Image
import torch.nn.functional as F

CATEGORIES = ["freshapple", "freshbanana", "freshorange",
              "rottenapple", "rottenbanana", "rottenorange"] # 6 categories

# This is a data model describing the output of the API
class Result(BaseModel):
    category: str
    confidence: float

# create the FastAPI app
app = FastAPI()

# debug message to show that the app is running
@app.get("/")
def read_root():
    return {"message": "API is running. Visit /docs for the swagger API."}

@app.post("/predict", response_model=Result)
async def predict(
    input_image: UploadFile = File(...),
    model: ResNet = Depends(load_model), # this will load the model and pass it to the function
    transforms: transforms.Compose = Depends(load_transform) # this will load the transform and pass it to the function
) -> Result:
    image = Image.open(io.BytesIO(await input_image.read())) # await is needed because read() is async
    if image.mode != "RGB":
        image = image.convert("RGB") # deal with grayscale images or images with alpha channel
    
    # Add the batch dimension for inference
    image = transforms(image).reshape(1, 3, 224, 224) # 1 batch dimension, 3 color channels, 224x224 image size

    model.eval() # drop off dropout layers and BatchNorm uses stats of training and not current batch
    with torch.inference_mode(): 
        # disable gradient calculation for inference (vs. eval dropping off dropout layers and BatchNorm uses stats of training and not current batch)
        # this saves memory and makes inference faster
        output = model(image) # forward pass
        category = CATEGORIES[output.argmax()] # get the index of the highest value in the output tensor and map it to the category
        confidence = F.softmax(output, dim=1).max().item() # get the confidence of the prediction
        return Result(category=category, confidence=confidence)

