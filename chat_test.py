import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import openai
from fuzzywuzzy import fuzz
import streamlit as st
import re
from pinecone import Pinecone
from typing import List, Dict
from upsert import get_lyrics_of_single_song, vectorize_single_song
import requests

# Initialize necessary APIs and settings
MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]  # Replace with your actual Pinecone index name

# Spotify developer credentials (Replace with your own or use environment variables)
client_id = st.secrets["SPOTIFY_CLIENT_ID"]
client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]

# Use Spotify's Client Credentials Flow for API access
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# OpenAI API initialization
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize Pinecone
pc = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"]
)
index = pc.Index(INDEX_NAME)


# Helper Functions

def get_embedding(text: str) -> List[float]:
    """Get an embedding for the given text using OpenAI's API."""
    response = openai.Embedding.create(input=[text], model=MODEL)
    return response['data'][0]['embedding']

def search_pinecone(query_vector: List[float], top_k: int = 5) -> List[Dict]:
    """Search the Pinecone index for similar vectors."""
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results['matches']

def process_query_with_gpt(query: str) -> str:
    """Process the user's query with GPT-4 to extract key search terms."""
    prompt = f"""
    Given the following question about a song, extract the key search terms that would be most relevant for finding the song in a database. Focus on unique identifiers like lyrics, artist names, or specific themes.

    Question: {query}

    Key search terms:
    """
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts key search terms from questions about songs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()

def search_spotify(query):
    """Search Spotify for song titles and artists."""
    results = sp.search(q=query, type='track', limit=10)  # Search for tracks, limit to top 10 results
    tracks = []
    for item in results['tracks']['items']:
        track_name = item['name']
        artist_name = item['artists'][0]['name']
        track_display = f"{track_name} by {artist_name}"
        tracks.append(track_display)
    return tracks

def get_embed_url(search_input, sp):
    """Retrieve the Spotify embed URL for a given song title and artist name."""
    if " by " in search_input:
        title, artist = search_input.split(" by ", 1)
    else:
        return None  # Invalid format, expected "TITLE by ARTIST"

    results = sp.search(q=f"track:{title} artist:{artist}", type='track', limit=5)
    for track in results['tracks']['items']:
        track_name = track['name'].lower()
        track_artists = ', '.join([a['name'].lower() for a in track['artists']])
        if title.lower() == track_name and artist.lower() in track_artists:
            track_id = track['id']
            embed_url = f"https://open.spotify.com/embed/track/{track_id}"
            return embed_url
    return None

def similarity_search(search_input):
    """Search for similar songs based on the lyrics vector of the input song."""
    lyrics = get_lyrics_of_single_song(search_input)
    res = openai.Embedding.create(input=lyrics[:8192], engine=MODEL)
    embedding = res['data'][0]['embedding']
    if not embedding:
        return None
    result = index.query(vector=embedding, top_k=10)
    suggestion_tuples = []
    for match in result['matches']:
        song_id = match['id']
        if fuzz.ratio(search_input.lower(), song_id.lower()) > 80:
            continue
        lyrics = get_lyrics_of_single_song(song_id)
        suggestion_tuples.append((song_id, lyrics))
        if len(suggestion_tuples) > 5:
            return suggestion_tuples  # Return the 5 most similar songs
    return None

def generate_song_description(input_lyrics, lyrics, input_song, suggestion_song):
    """Generate a witty, creative one-sentence description for given song lyrics using GPT-3.5."""
    messages = [
        {"role": "system", "content": "You are an AI music reviewer that compares the lyrics of two songs and writes a 1 sentence description of why the songs are similar for a website's music catalog."},
        {"role": "user", "content": f"Create a 1 sentence description of why these songs' lyrics are similar. Use the input song name {input_song} and suggested song name {suggestion_song} in the sentence:\n\n{input_lyrics}{lyrics}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=50,
        temperature=0.7,
        n=1
    )
    description = response['choices'][0]['message']['content'].strip()
    return description



import streamlit as st

# Set background color to light green using Streamlit's style function
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ccffcc;
    }
    .header-container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-container img {
        width: 100px; /* Adjust size as needed */
        margin-right: 10px;
    }
    .spaced-container {
        padding-left: 50px;
        padding-right: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display images at the top using columns for alignment
top_left_col, top_middle_col, top_right_col = st.columns([1, 1, 1])

# Display the images in each column at the top
with top_left_col:
    st.image("Pics/spotifylogo.png", width=100, caption="Spotify")

with top_middle_col:
    st.image("Pics/openailogo.png", width=100, caption="OpenAI")

with top_right_col:
    st.image("Pics/pineconelogo.jpg", width=100, caption="Pinecone")

# Increase the spacing between the columns by adjusting column ratios
left_col, right_col = st.columns([1, 1])  # Adjust column ratio to add spacing on the sides

# LEFT COLUMN - Similar Song Search Section
with left_col:
    st.header("Lyric Similarity Search")
    st.markdown('<div class="spaced-container">', unsafe_allow_html=True)  # Add left padding
    search_input = st.text_input("Type a song title or artist name:")

    if search_input:
        suggestions = search_spotify(search_input)
        col1, col2 = st.columns([3, 1])
        selected_song = col1.selectbox("Suggestions:", suggestions, key="dropdown")
        search_button = col2.button("Search")

        if search_button and selected_song:
            st.write(f"You selected: {selected_song}")
            input_lyrics = get_lyrics_of_single_song(selected_song)

            if input_lyrics:
                similar_songs = similarity_search(selected_song)
                if similar_songs:
                    for tuple_item in similar_songs:
                        song, lyrics = tuple_item
                        st.write(f"Similar song found: **{song}**")
                        spotify_embed_url = get_embed_url(song, sp)
                        st.components.v1.iframe(spotify_embed_url, width=300, height=80)
                        description = generate_song_description(input_lyrics, lyrics, selected_song, song)
                        st.write(description)
                        st.write(f"**Lyrics**: {lyrics}")
                else:
                    st.error("No similar songs found.")
    st.markdown('</div>', unsafe_allow_html=True)  # Close the div container

# RIGHT COLUMN - Chatbot-based Lyric Search Section
with right_col:
    st.header("Lyric-based Song Finder")
    st.markdown('<div class="spaced-container">', unsafe_allow_html=True)  # Add right padding
    query = st.text_input("Enter any phrase to find songs about that topic:")

    if st.button("Search Song by Lyrics"):
        if query:
            with st.spinner("Processing your question..."):
                processed_query = process_query_with_gpt(query)
                st.write(f"Processed search terms: {processed_query}")
                query_vector = get_embedding(processed_query)
                results = search_pinecone(query_vector)
                st.subheader("Search Results:")

                for i, result in enumerate(results, 1):
                    st.write(f"{i}. {result['id']} (Score: {result['score']:.4f})")
                    if 'metadata' in result:
                        st.write(f"   Artist: {result['metadata'].get('artist', 'Unknown')}")
                        st.write(f"   Title: {result['metadata'].get('title', 'Unknown')}")
                        st.write(f"   Album: {result['metadata'].get('album', 'Unknown')}")
                    st.write("---")
    st.markdown('</div>', unsafe_allow_html=True)  # Close the div container.
