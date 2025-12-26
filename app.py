from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import io
import os
import uuid
from datetime import datetime

# Initialize FastAPI
app = FastAPI(title="Real-vs-AI Image Detector", version="1.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load model 
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 2)

# Load saved trained checkpoint
checkpoint = torch.load(
    r"C:\Users\ASHU\Videos\AI-ML Engineer projects\real-vs-ai-image-detector\model\best_efficientnet_b0.pth",
    map_location=device
)

# Load only the model weights
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Move model to the device and set to evaluation mode
model.to(device)
model.eval()

# image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Endpoint 
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint image prediction
@app.post("/predict/")
async def predict_image(request: Request, file: UploadFile = File(...)):
    try:
        # filename
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read image for processing
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = float(probabilities[0][pred] * 100)
        
        # Map prediction
        label_map = {0: "AI Generated", 1: "Real Photo"}
        
        # Get prediction color
        colors = {
            "AI Generated": "#ef4444",  # Red for AI
            "Real Photo": "#10b981"     # Green for Real
        }
        
        result = {
            "filename": unique_filename,
            "original_name": file.filename,
            "prediction": label_map[pred],
            "confidence": round(confidence, 2),
            "color": colors[label_map[pred]],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "show_result": True
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Error: {str(e)}",
                "show_error": True
            }
        )

# Endpoint for API-only prediction 
@app.post("/api/predict/")
async def api_predict_image(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = float(probabilities[0][pred] * 100)
        
        label_map = {0: "AI Generated", 1: "Real Photo"}
        
        return JSONResponse({
            "filename": file.filename,
            "prediction": label_map[pred],
            "confidence": round(confidence, 2),
            "status": "success"
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "error"}, status_code=500)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded", "device": device}