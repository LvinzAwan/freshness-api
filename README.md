---
title: Freshness Detection API
emoji: 🥬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🥬 Freshness Detection API

AI-powered freshness detection system untuk buah, sayur, dan daging.

## Usage

### Single Prediction
```bash
curl -X POST https://USERNAME-freshness-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_string", "category": "buah"}'