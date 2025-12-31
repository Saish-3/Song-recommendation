Song Recommendation System

ML-powered Recommendation API | FastAPI | scikit-learn

A production-ready content-based song recommendation system demonstrating how machine learning models are transformed into scalable backend APIs under real-world constraints.

🔗 Live API: https://song-recommendation-5m1c.onrender.com

📌 Built during: DevTown AI Recommendation Bootcamp (MSME – Startup India)

🚀 Why this project stands out

✔ Built a full ML pipeline, not just a notebook
✔ Converted ML logic into a REST API
✔ Deployed with memory-aware architecture
✔ Handled real cloud constraints (512 MB RAM)
✔ Designed for production stability, not demos

This project focuses on how ML systems are actually deployed, not just trained.
How it works (High-level)

Song titles + artists are combined into text features

Text is vectorized using TF-IDF

Similarity is computed using Cosine Similarity

Fuzzy matching handles imperfect user input

Recommendations are returned via a FastAPI endpoint

Type: Content-Based Recommendation System (NLP)

🛠️ Tech Stack

Python

FastAPI – backend API

scikit-learn

TF-IDF Vectorizer

Cosine Similarity

pandas – data processing

difflib – fuzzy matching

Uvicorn – ASGI server

Render – cloud deployment
API Endpoints
GET /

Health check endpoint.

{
  "status": "API running",
  "dataset_loaded": false
}

POST /recommend_songs

Request

{
  "song": "Believer"
}


Response (local execution)

{
  "matched_song": "Believer",
  "recommendations": ["Thunder", "Radioactive", "Demons", ...]
}

⚠️ Deployment Architecture (Important)

The full dataset (songs.csv, ~69 MB) is intentionally excluded from the deployed server.

Why?

Render Free Tier → 512 MB RAM

Loading CSV + TF-IDF + similarity matrix exceeds memory

Naive deployment would crash the service

Solution implemented:

Lazy loading of ML components

Graceful handling when dataset is unavailable

API remains live and stable in production

✅ Full functionality works locally
✅ Deployed API demonstrates real ML backend design

This mirrors industry-standard ML service architecture.
⭐ If you found this useful, consider starring the repo

It helps others discover practical ML deployment examples.
