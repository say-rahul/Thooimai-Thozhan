import os
import requests
from io import BytesIO
from PIL import Image

import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

app = FastAPI(
    title="Thooimai Waste AI",
    version="1.0.0",
)

# ============================================================
# CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client | None = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "karthikeya09/smart_image_recognation"
MODEL_FILE = "best_model.pth"

MODEL_CLASSES = [
    "glass",
    "metal",
    "non-recyclable",
    "organic",
    "paper",
    "plastic",
]

APP_CATEGORIES = {
    "organic": "Wet Waste",
    "plastic": "Plastic",
    "paper": "Paper",
    "metal": "Metal",
    "glass": "Glass",
    "non-recyclable": "Other",
}

# Transform pipeline matching MobileNetV2 training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model = None

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class WasteClassifier(torch.nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(
                self.backbone.last_channel,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.backbone(x)

# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    global model

    if model is not None:
        return model

    print("============================================")
    print("Loading Thooimai Waste AI model...")
    print("============================================")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    print("Model downloaded from Hugging Face:", model_path)

    net = WasteClassifier(num_classes=len(MODEL_CLASSES))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict, strict=True)
    net.eval()

    model = net

    print("============================================")
    print("Waste AI model loaded successfully.")
    print("Classes:", MODEL_CLASSES)
    print("============================================")

    return model

# Core prediction helper function
def run_model_inference(image: Image.Image):
    net = load_model()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = net(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, index = torch.max(probabilities, dim=0)

    predicted_material = MODEL_CLASSES[index.item()]
    confidence_value = float(confidence.item())
    application_category = APP_CATEGORIES.get(predicted_material, "Other")

    predictions = {
        MODEL_CLASSES[i]: round(float(probabilities[i].item()), 4)
        for i in range(len(MODEL_CLASSES))
    }

    return {
        "predicted_material": predicted_material,
        "category": application_category,
        "confidence": round(confidence_value * 100, 2),
        "confidence_raw": round(confidence_value, 4),
        "predictions": predictions,
    }

# ============================================================
# HEALTH & ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Thooimai Waste AI",
        "status": "online",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None,
    }

# ============================================================
# PROCESS RECENT TICKET FROM SUPABASE
# ============================================================

@app.post("/process-latest-ticket")
def process_latest_ticket():
    """
    Fetches the latest ticket with ai_verification_status = 'pending',
    downloads its photo from Supabase Storage, runs AI classification,
    and updates the database row.
    """
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client not configured.")

    try:
        # Fetch latest pending ticket
        response = supabase.table("pickup_tickets") \
            .select("*") \
            .eq("ai_verification_status", "pending") \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        if not response.data or len(response.data) == 0:
            return {"success": True, "message": "No pending tickets found."}

        ticket = response.data[0]
        ticket_id = ticket["id"]
        photo_path = ticket.get("waste_photo_path")
        user_category = ticket.get("waste_category")

        if not photo_path:
            raise HTTPException(status_code=400, detail="Ticket missing waste_photo_path")

        # Get signed photo URL from Supabase Storage
        signed_res = supabase.storage.from_("waste-photos").create_signed_url(photo_path, 120)
        signed_url = signed_res.get("signedUrl")

        if not signed_url:
            raise HTTPException(status_code=400, detail="Could not generate signed photo URL")

        # Download image into memory
        img_bytes = requests.get(signed_url).content
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Classify with PyTorch model
        ai_res = run_model_inference(image)

        # Match verification
        is_match = ai_res["category"].lower() == user_category.lower()
        verification_status = "verified" if is_match else "mismatch"

        # Update record in Supabase
        update_payload = {
            "ai_predicted_category": ai_res["category"],
            "ai_confidence": ai_res["confidence_raw"],
            "ai_verification_status": verification_status,
        }

        supabase.table("pickup_tickets") \
            .update(update_payload) \
            .eq("id", ticket_id) \
            .execute()

        # Log prediction
        try:
            supabase.table("waste_predictions").insert({
                "predicted_material": ai_res["predicted_material"],
                "category": ai_res["category"],
                "confidence": ai_res["confidence"],
                "predictions": ai_res["predictions"],
                "image_filename": photo_path.split("/")[-1],
            }).execute()
        except Exception as log_err:
            print("Waste prediction logging failed:", log_err)

        return {
            "success": True,
            "ticket_id": ticket_id,
            "ai_predicted_category": ai_res["category"],
            "ai_confidence": ai_res["confidence"],
            "ai_verification_status": verification_status,
        }

    except Exception as exc:
        print("Ticket processing error:", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ============================================================
# SUPABASE WEBHOOK LISTENER
# ============================================================

@app.post("/webhook/classify-ticket")
async def classify_ticket_webhook(payload: dict = Body(...)):
    """
    Called by Supabase Database Webhooks whenever a new row is inserted into pickup_tickets.
    """
    record = payload.get("record", {})
    ticket_id = record.get("id")
    photo_path = record.get("waste_photo_path")
    user_category = record.get("waste_category")

    if not ticket_id or not photo_path:
        raise HTTPException(status_code=400, detail="Missing ticket_id or photo_path in payload")

    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client not configured")

    try:
        # Download photo via signed URL
        signed_res = supabase.storage.from_("waste-photos").create_signed_url(photo_path, 120)
        signed_url = signed_res.get("signedUrl")

        if not signed_url:
            raise HTTPException(status_code=400, detail="Failed to get signed URL")

        img_bytes = requests.get(signed_url).content
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # AI Prediction
        ai_res = run_model_inference(image)

        is_match = ai_res["category"].lower() == str(user_category).lower()
        verification_status = "verified" if is_match else "mismatch"

        # Update pickup_tickets
        supabase.table("pickup_tickets").update({
            "ai_predicted_category": ai_res["category"],
            "ai_confidence": ai_res["confidence_raw"],
            "ai_verification_status": verification_status,
        }).eq("id", ticket_id).execute()

        return {
            "success": True,
            "ticket_id": ticket_id,
            "verification_status": verification_status,
            "predicted_category": ai_res["category"],
        }

    except Exception as err:
        print("Webhook classification failed:", err)

        supabase.table("pickup_tickets").update({
            "ai_verification_status": "failed"
        }).eq("id", ticket_id).execute()

        raise HTTPException(status_code=500, detail=str(err))

# ============================================================
# DIRECT MULTIPART FILE PREDICT (LEGACY/DIRECT)
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        ai_res = run_model_inference(image)

        if supabase is not None:
            try:
                supabase.table("waste_predictions").insert({
                    "predicted_material": ai_res["predicted_material"],
                    "category": ai_res["category"],
                    "confidence": ai_res["confidence"],
                    "predictions": ai_res["predictions"],
                    "image_filename": file.filename,
                }).execute()
            except Exception as db_error:
                print("Supabase log insert failed:", db_error)

        return {
            "success": True,
            "predicted_material": ai_res["predicted_material"],
            "category": ai_res["category"],
            "confidence": ai_res["confidence"],
            "predictions": ai_res["predictions"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("Prediction error:", exc)
        raise HTTPException(status_code=500, detail=f"AI prediction failed: {str(exc)}")

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    try:
        load_model()
        print("============================================")
        print("Waste AI model pre-loaded successfully.")
        print("============================================")
    except Exception as exc:
        print("Model startup load failed:", exc)
