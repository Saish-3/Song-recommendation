from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import difflib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- APP SETUP --------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GLOBALS (LAZY LOADED) --------------------

DATA_PATH = "songs.csv"

song_data = None
similarity = None
songs_list = None
vectorizer = None

# -------------------- LOAD DATA ONLY WHEN NEEDED --------------------

def load_model():
    global song_data, similarity, songs_list, vectorizer

    # If already loaded, do nothing
    if song_data is not None:
        return

    # If dataset not present (Render)
    if not os.path.exists(DATA_PATH):
        return

    song_data = pd.read_csv(
        DATA_PATH,
        engine="python",
        on_bad_lines="skip"
    )

    features = ["song", "artist"]
    for feature in features:
        song_data[feature] = song_data[feature].fillna("")

    combined_features = song_data["song"] + " " + song_data["artist"]

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    similarity = cosine_similarity(feature_vectors)
    songs_list = song_data["song"].tolist()


class SongRequest(BaseModel):
    song: str


@app.get("/")
def root():
    return {
        "status": "API running",
        "dataset_loaded": os.path.exists(DATA_PATH)
    }

@app.post("/recommend_songs")
def recommend_songs(request: SongRequest):
    load_model()

    if song_data is None:
        return {
            "error": "Dataset not available on server",
            "message": "Recommendations work locally, dataset removed for Render free tier"
        }

    song_name = request.song

    close_matches = difflib.get_close_matches(song_name, songs_list)
    if not close_matches:
        return {"error": "Song not found"}

    close_match = close_matches[0]
    index = song_data[song_data.song == close_match].index[0]

    similarity_scores = list(enumerate(similarity[index]))
    sorted_similar_songs = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    for i, score in sorted_similar_songs[1:21]:
        recommendations.append(song_data.iloc[i]["song"])

    return {
        "matched_song": close_match,
        "recommendations": recommendations
    }

