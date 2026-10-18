import os
import requests
from io import BytesIO

from PIL import Image, UnidentifiedImageError

import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client


app = FastAPI(
    title="Thooimai Waste AI",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SUPABASE CONFIGURATION
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

    print("Loading Thooimai Waste AI model...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    print("Model downloaded from Hugging Face:", model_path)

    net = WasteClassifier(num_classes=len(MODEL_CLASSES))

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

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

    print("Waste AI model loaded successfully.")
    print("Classes:", MODEL_CLASSES)

    return model


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_model_inference(image: Image.Image):
    net = load_model()

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = net(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, index = torch.max(probabilities, dim=0)

    predicted_material = MODEL_CLASSES[index.item()]
    confidence_value = float(confidence.item())

    application_category = APP_CATEGORIES.get(
        predicted_material,
        "Other",
    )

    predictions = {
        MODEL_CLASSES[i]: round(
            float(probabilities[i].item()),
            4,
        )
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
# DOWNLOAD IMAGE FROM SUPABASE STORAGE
# ============================================================

def download_ticket_image(photo_path: str) -> Image.Image:
    if supabase is None:
        raise RuntimeError("Supabase client is not configured.")

    if not photo_path:
        raise RuntimeError("Ticket is missing waste_photo_path.")

    signed_response = (
        supabase.storage
        .from_("waste-photos")
        .create_signed_url(photo_path, 300)
    )

    if isinstance(signed_response, dict):
        signed_url = signed_response.get("signedUrl")
    else:
        signed_url = getattr(
            signed_response,
            "signed_url",
            None,
        )

    if not signed_url:
        raise RuntimeError("Could not generate signed photo URL.")

    image_response = requests.get(
        signed_url,
        timeout=30,
    )
    image_response.raise_for_status()

    if not image_response.content:
        raise RuntimeError("Downloaded waste photo is empty.")

    try:
        return Image.open(
            BytesIO(image_response.content)
        ).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise RuntimeError(
            "The uploaded file is not a valid image."
        ) from error


# ============================================================
# CLASSIFY AND SAVE A SPECIFIC TICKET
# ============================================================

def classify_and_save_ticket(
    ticket_id: str,
    photo_path: str,
    user_category: str,
):
    if supabase is None:
        raise RuntimeError("Supabase client is not configured.")

    image = download_ticket_image(photo_path)
    ai_result = run_model_inference(image)

    predicted_category = ai_result["category"]

    is_match = (
        predicted_category.strip().casefold()
        == str(user_category).strip().casefold()
    )

    verification_status = (
        "verified" if is_match else "mismatch"
    )

    update_result = (
        supabase.table("pickup_tickets")
        .update({
            "ai_predicted_category": predicted_category,
            "ai_confidence": ai_result["confidence_raw"],
            "ai_verification_status": verification_status,
        })
        .eq("id", ticket_id)
        .execute()
    )

    if not update_result.data:
        raise RuntimeError(
            "AI inference completed, but the ticket update "
            "could not be confirmed."
        )

    # Prediction history is supplementary. A logging failure
    # must not invalidate a successfully saved ticket result.
    try:
        supabase.table("waste_predictions").insert({
            "predicted_material": ai_result["predicted_material"],
            "category": predicted_category,
            "confidence": ai_result["confidence"],
            "predictions": ai_result["predictions"],
            "image_filename": photo_path.split("/")[-1],
        }).execute()
    except Exception as log_error:
        print("Waste prediction logging failed:", log_error)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "predicted_material": ai_result["predicted_material"],
        "predicted_category": predicted_category,
        "confidence": ai_result["confidence"],
        "confidence_raw": ai_result["confidence_raw"],
        "verification_status": verification_status,
        "predictions": ai_result["predictions"],
    }


# ============================================================
# ROOT AND HEALTH CHECK
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
# PROCESS THE LATEST PENDING TICKET
# ============================================================

@app.post("/process-latest-ticket")
def process_latest_ticket():
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase client not configured.",
        )

    try:
        response = (
            supabase.table("pickup_tickets")
            .select("*")
            .eq("ai_verification_status", "pending")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if not response.data:
            return {
                "success": True,
                "message": "No pending tickets found.",
            }

        ticket = response.data[0]

        result = classify_and_save_ticket(
            ticket_id=ticket["id"],
            photo_path=ticket.get("waste_photo_path"),
            user_category=ticket.get("waste_category"),
        )

        return result

    except HTTPException:
        raise
    except Exception as error:
        print("Latest ticket processing failed:", error)

        raise HTTPException(
            status_code=500,
            detail=f"Ticket processing failed: {str(error)}",
        )


# ============================================================
# SUPABASE WEBHOOK: CLASSIFY THE INSERTED TICKET
# ============================================================

@app.post("/webhook/classify-ticket")
@app.post("/webhook/classify-ticket/")
async def classify_ticket_webhook(
    payload: dict = Body(...),
):
    """
    Expected request body:

    {
        "record": {
            "id": "ticket UUID",
            "waste_category": "Wet Waste",
            "waste_photo_path": "user-id/ticket-id/original.jpg"
        }
    }

    This also matches the record-shaped payload sent by
    the updated React Native pickup service.
    """

    record = payload.get("record") or {}

    ticket_id = record.get("id")
    photo_path = record.get("waste_photo_path")
    user_category = record.get("waste_category")

    if not ticket_id or not photo_path or not user_category:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing ticket ID, waste_photo_path, "
                "or waste_category in record."
            ),
        )

    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase client is not configured.",
        )

    try:
        # Confirm that the ticket exists and use the current
        # database values instead of blindly trusting the payload.
        ticket_response = (
            supabase.table("pickup_tickets")
            .select(
                "id, waste_category, waste_photo_path, "
                "ai_verification_status"
            )
            .eq("id", ticket_id)
            .maybe_single()
            .execute()
        )

        ticket = ticket_response.data

        if not ticket:
            raise HTTPException(
                status_code=404,
                detail="Pickup ticket not found.",
            )

        actual_photo_path = ticket.get("waste_photo_path")
        actual_category = ticket.get("waste_category")

        if not actual_photo_path or not actual_category:
            raise HTTPException(
                status_code=400,
                detail="Pickup ticket is missing its photo or category.",
            )

        result = classify_and_save_ticket(
            ticket_id=ticket["id"],
            photo_path=actual_photo_path,
            user_category=actual_category,
        )

        return result

    except HTTPException:
        raise
    except Exception as error:
        print(
            f"Webhook classification failed for ticket "
            f"{ticket_id}: {error}"
        )

        # Preserve the pickup request. Only change the AI status.
        try:
            (
                supabase.table("pickup_tickets")
                .update({"ai_verification_status": "failed"})
                .eq("id", ticket_id)
                .execute()
            )
        except Exception as update_error:
            print(
                "Could not save AI failure status:",
                update_error,
            )

        raise HTTPException(
            status_code=500,
            detail=f"AI classification failed: {str(error)}",
        )


# ============================================================
# DIRECT IMAGE PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        try:
            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            ) from error

        ai_result = run_model_inference(image)

        if supabase is not None:
            try:
                supabase.table("waste_predictions").insert({
                    "predicted_material": ai_result["predicted_material"],
                    "category": ai_result["category"],
                    "confidence": ai_result["confidence"],
                    "predictions": ai_result["predictions"],
                    "image_filename": file.filename,
                }).execute()
            except Exception as database_error:
                print(
                    "Supabase prediction logging failed:",
                    database_error,
                )

        return {
            "success": True,
            "predicted_material": ai_result["predicted_material"],
            "category": ai_result["category"],
            "confidence": ai_result["confidence"],
            "confidence_raw": ai_result["confidence_raw"],
            "predictions": ai_result["predictions"],
        }

    except HTTPException:
        raise
    except Exception as error:
        print("Prediction error:", error)

        raise HTTPException(
            status_code=500,
            detail=f"AI prediction failed: {str(error)}",
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    try:
        load_model()
        print("Waste AI model pre-loaded successfully.")
    except Exception as error:
        # Keep the service up so /health can report its state.
        # Inference will retry model loading when requested.
        print("Model startup load failed:", error)


# ============================================================
# RENDER ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
