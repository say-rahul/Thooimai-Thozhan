# Thooimai Waste AI

Basic waste-material classification API for the Thooimai mobile app.

## Classes

The base MobileNetV2 model recognizes:

- organic
- plastic
- paper
- metal
- glass
- non-recyclable

Application mapping:

- organic -> Wet Waste
- plastic -> Plastic
- paper -> Paper
- metal -> Metal
- glass -> Glass
- non-recyclable -> Other

This is a prototype classifier. It does not reliably identify construction waste,
all e-waste, or mixed household waste as separate classes.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Render

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The model is downloaded from Hugging Face on startup.

## Endpoint

```text
POST /predict
```

Multipart field:

```text
file
```

Example response:

```json
{
  "success": true,
  "predicted_material": "organic",
  "category": "Wet Waste",
  "confidence": 92.31,
  "predictions": {
    "glass": 0.001,
    "metal": 0.002,
    "non-recyclable": 0.01,
    "organic": 0.9231,
    "paper": 0.02,
    "plastic": 0.0439
  }
}
```
