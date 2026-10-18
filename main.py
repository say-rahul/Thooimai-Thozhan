import os
import requests
from io import BytesIO
from threading import Lock

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from fastai.vision.all import PILImage
from huggingface_hub import from_pretrained_fastai

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client


app = FastAPI(
    title="Thooimai Waste AI",
    version="2.1.0",
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

MODEL_REPO = "C0mpl1cated3/trash_classifier_v2"

APP_CATEGORIES = {
    "cardboard": "Paper",
    "e-waste": "Other",
    "glass": "Glass",
    "metal": "Metal",
    "organic": "Wet Waste",
    "paper": "Paper",
    "plastic": "Plastic",
    "textile": "Other",
}

# Initial threshold only. Validate this against labelled images.
MIN_CONFIDENCE = 0.60

model = None
model_lock = Lock()


# ============================================================
# LAZY MODEL LOADING
# ============================================================

def load_model():
    global model

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        print(f"Loading waste model: {MODEL_REPO}")

        try:
            learner = from_pretrained_fastai(MODEL_REPO)
            learner.model.eval()

            labels = [
                str(label).strip().lower()
                for label in learner.dls.vocab
            ]

            if not labels:
                raise RuntimeError("The model returned no class labels.")

            missing = [
                label for label in labels
                if label not in APP_CATEGORIES
            ]

            if missing:
                raise RuntimeError(
                    "Missing category mappings for: "
                    + ", ".join(missing)
                )

            model = learner

            print("Waste model loaded successfully.")
            print("Model classes:", labels)

            return model

        except Exception as error:
            print(f"Model loading failed: {error}")
            raise RuntimeError(
                "Could not load the Hugging Face waste model. "
                "Check the repository loading instructions and Render logs."
            ) from error


# ============================================================
# IMAGE INFERENCE
# ============================================================

def run_model_inference(image: Image.Image):
    learner = load_model()

    image = ImageOps.exif_transpose(image).convert("RGB")
    pil_image = PILImage.create(image)

    with torch.inference_mode():
        predicted_label, predicted_index, probabilities = (
            learner.predict(pil_image)
        )

    material = str(predicted_label).strip().lower()

    classes = [
        str(label).strip().lower()
        for label in learner.dls.vocab
    ]

    if material not in classes:
        raise RuntimeError(
            f"Model returned an unknown label: {material}"
        )

    if material not in APP_CATEGORIES:
        raise RuntimeError(
            f"No application category mapping for: {material}"
        )

    confidence = float(probabilities[int(predicted_index)])

    predictions = {
        label: round(float(probabilities[index]), 4)
        for index, label in enumerate(classes)
    }

    needs_review = confidence < MIN_CONFIDENCE

    return {
        "predicted_material": material,
        "category": APP_CATEGORIES[material],
        "confidence": round(confidence * 100, 2),
        "confidence_raw": round(confidence, 4),
        "predictions": predictions,
        "needs_review": needs_review,
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
        signed_url = (
            signed_response.get("signedUrl")
            or signed_response.get("signedURL")
        )
    else:
        signed_url = getattr(
            signed_response,
            "signed_url",
            None,
        )

    if not signed_url:
        raise RuntimeError("Could not generate signed photo URL.")

    response = requests.get(signed_url, timeout=30)
    response.raise_for_status()

    if not response.content:
        raise RuntimeError("Downloaded waste photo is empty.")

    try:
        return Image.open(
            BytesIO(response.content)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        raise RuntimeError(
            "The uploaded file is not a valid image."
        ) from error


# ============================================================
# CLASSIFY AND SAVE TICKET
# ============================================================

def classify_and_save_ticket(
    ticket_id: str,
    photo_path: str,
    user_category: str,
):
    if supabase is None:
        raise RuntimeError("Supabase client is not configured.")

    image = download_ticket_image(photo_path)
    result = run_model_inference(image)

    predicted_category = result["category"]

    is_match = (
        predicted_category.strip().casefold()
        == str(user_category).strip().casefold()
    )

    if result["needs_review"]:
        verification_status = "pending"
    else:
        verification_status = (
            "verified" if is_match else "mismatch"
        )

    update_result = (
        supabase.table("pickup_tickets")
        .update({
            "ai_predicted_category": predicted_category,
            "ai_confidence": result["confidence_raw"],
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

    # Prediction history is supplementary.
    try:
        supabase.table("waste_predictions").insert({
            "predicted_material": result["predicted_material"],
            "category": predicted_category,
            "confidence": result["confidence"],
            "predictions": result["predictions"],
            "image_filename": photo_path.split("/")[-1],
        }).execute()

    except Exception as error:
        print("Prediction logging failed:", error)

    return {
        "success": True,
        "ticket_id": ticket_id,
        "predicted_material": result["predicted_material"],
        "predicted_category": predicted_category,
        "confidence": result["confidence"],
        "confidence_raw": result["confidence_raw"],
        "verification_status": verification_status,
        "needs_review": result["needs_review"],
        "predictions": result["predictions"],
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
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():
    try:
        learner = load_model()

        return {
            "model": MODEL_REPO,
            "classes": [
                str(label)
                for label in learner.dls.vocab
            ],
            "status": "loaded",
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


# ============================================================
# PROCESS LATEST PENDING TICKET
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

        return classify_and_save_ticket(
            ticket_id=ticket["id"],
            photo_path=ticket.get("waste_photo_path"),
            user_category=ticket.get("waste_category"),
        )

    except HTTPException:
        raise

    except Exception as error:
        print("Latest ticket processing failed:", error)

        raise HTTPException(
            status_code=500,
            detail=f"Ticket processing failed: {str(error)}",
        ) from error


# ============================================================
# SUPABASE WEBHOOK
# ============================================================

@app.post("/webhook/classify-ticket")
@app.post("/webhook/classify-ticket/")
async def classify_ticket_webhook(
    payload: dict = Body(...),
):
    record = payload.get("record") or {}
    ticket_id = record.get("id")

    if not ticket_id:
        raise HTTPException(
            status_code=400,
            detail="Missing ticket ID in webhook record.",
        )

    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase client is not configured.",
        )

    try:
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

        photo_path = ticket.get("waste_photo_path")
        user_category = ticket.get("waste_category")

        if not photo_path or not user_category:
            raise HTTPException(
                status_code=400,
                detail="Pickup ticket is missing its photo or category.",
            )

        return classify_and_save_ticket(
            ticket_id=ticket["id"],
            photo_path=photo_path,
            user_category=user_category,
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"Webhook classification failed for {ticket_id}: {error}"
        )

        try:
            (
                supabase.table("pickup_tickets")
                .update({"ai_verification_status": "failed"})
                .eq("id", ticket_id)
                .execute()
            )
        except Exception as update_error:
            print("Could not save AI failure status:", update_error)

        raise HTTPException(
            status_code=500,
            detail=f"AI classification failed: {str(error)}",
        ) from error


# ============================================================
# DIRECT IMAGE PREDICTION
# ============================================================

@app.post("/predict")
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
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

        result = run_model_inference(image)

        if supabase is not None:
            try:
                supabase.table("waste_predictions").insert({
                    "predicted_material": result["predicted_material"],
                    "category": result["category"],
                    "confidence": result["confidence"],
                    "predictions": result["predictions"],
                    "image_filename": file.filename,
                }).execute()

            except Exception as error:
                print("Prediction logging failed:", error)

        return {
            "success": True,
            "model": MODEL_REPO,
            "predicted_material": result["predicted_material"],
            "category": result["category"],
            "confidence": result["confidence"],
            "confidence_raw": result["confidence_raw"],
            "needs_review": result["needs_review"],
            "predictions": result["predictions"],
        }

    except HTTPException:
        raise

    except Exception as error:
        print("Prediction error:", error)

        raise HTTPException(
            status_code=503,
            detail=f"AI prediction failed: {str(error)}",
        ) from error


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    # Do not download the model during startup.
    # The first /predict or /model-info request loads it.
    print("Thooimai Waste AI API starting...")
    print(f"Model will load on first request: {MODEL_REPO}")


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
