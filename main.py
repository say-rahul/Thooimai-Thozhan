from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

import os
from supabase import create_client, Client


app = FastAPI(
    title="Thooimai Waste AI",
    version="1.0.0",
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
# MODEL
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
# LOAD MODEL
# ============================================================

def load_model():
    global model

    if model is not None:
        return model

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    net = models.mobilenet_v2(weights=None)

    net.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1280, 6),
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    net.load_state_dict(state_dict)

    net.eval()

    model = net

    return model


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Thooimai Waste AI",
        "status": "online",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "supabase_connected": supabase is not None,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")


        # ----------------------------------------------------
        # Load AI model
        # ----------------------------------------------------

        net = load_model()


        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        tensor = transform(image).unsqueeze(0)


        # ----------------------------------------------------
        # AI prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = net(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]


        confidence, index = torch.max(
            probabilities,
            dim=0,
        )


        predicted_material = MODEL_CLASSES[
            index.item()
        ]


        confidence_value = float(
            confidence.item()
        )


        # ----------------------------------------------------
        # All predictions
        # ----------------------------------------------------

        predictions = {
            MODEL_CLASSES[i]: round(
                float(probabilities[i].item()),
                4,
            )
            for i in range(len(MODEL_CLASSES))
        }


        # ----------------------------------------------------
        # Application category
        # ----------------------------------------------------

        application_category = APP_CATEGORIES[
            predicted_material
        ]


        # ----------------------------------------------------
        # SAVE TO SUPABASE
        # ----------------------------------------------------

        if supabase is not None:

            try:

                prediction_data = {
                    "predicted_material": predicted_material,
                    "category": application_category,
                    "confidence": round(
                        confidence_value * 100,
                        2,
                    ),
                    "predictions": predictions,
                    "image_filename": file.filename,
                }

                supabase.table(
                    "waste_predictions"
                ).insert(
                    prediction_data
                ).execute()

                print(
                    "Prediction saved to Supabase:",
                    prediction_data,
                )

            except Exception as db_error:

                # Don't fail the AI prediction just because
                # database logging failed.

                print(
                    "Supabase insert failed:",
                    db_error,
                )


        # ----------------------------------------------------
        # Return result to mobile app
        # ----------------------------------------------------

        return {
            "success": True,

            "predicted_material":
                predicted_material,

            "category":
                application_category,

            "confidence":
                round(
                    confidence_value * 100,
                    2,
                ),

            "predictions":
                predictions,
        }


    except HTTPException:
        raise


    except Exception as exc:

        print(
            "Prediction error:",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=f"AI prediction failed: {str(exc)}",
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    try:

        load_model()

        print(
            "Waste AI model loaded successfully."
        )

    except Exception as exc:

        print(
            "Model startup load failed:",
            exc,
        )
