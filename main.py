from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
song_data = pd.read_csv(
    "songs.csv",
    engine="python",
    on_bad_lines="skip"
)
features = ['song', 'artist']
for feature in features:
    song_data[feature] = song_data[feature].fillna('')
c_features = song_data['song']+''+song_data['artist']
vectorizer = TfidfVectorizer()
f_vectorizer = vectorizer.fit_transform(c_features)
similar = cosine_similarity(f_vectorizer)
songs_list = song_data['song'].to_list()
class song_request(BaseModel):
    song: str
@app.get("/")
def root():
    return{"message": "Welcome to the Song Recommendation API"}
@app.post("/recommend_songs/")
def recommend_songs(request: song_request):
    song_name = request.song
    find_match = difflib.get_close_matches(song_name,songs_list)
    close_match = find_match[0]
    song_index = song_data[song_data.song == close_match].index[0]
    similarity_score = list(enumerate(similar[song_index]))
    sort_similar_songs = sorted(similarity_score,key=lambda x:x[1], reverse = True)
    recommendations = []
    for i , song in enumerate(sort_similar_songs[1:20]):
        index= song[0]
        song = song_data[song_data.index == index]['song'].values[0]
        recommendations.append(song)
    return{"matched_song": close_match, "recommendations": recommendations}
