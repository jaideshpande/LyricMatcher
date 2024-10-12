import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import Counter
import time
import openai
import os
from pinecone import Pinecone, ServerlessSpec
from fuzzywuzzy import fuzz
import streamlit as st
import langdetect
from langdetect import detect
import re

from upsert import get_lyrics_of_single_song, vectorize_single_song





# Initialize necessary APIs and settings
MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]  # Replace with your actual Pinecone index name

# Spotify developer credentials (Replace with your own or use environment variables)
client_id = st.secrets["SPOTIFY_CLIENT_ID"]
client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
redirect_uri = 'http://localhost:3000/callback'
scope = 'playlist-read-collaborative playlist-read-private user-modify-playback-state user-read-playback-state'
auth_manager = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope)
sp = spotipy.Spotify(auth_manager=auth_manager)

# OpenAI API initialization
openai.api_key = st.secrets["OPENAI_API_KEY"]

pc = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"]
)
index=pc.Index(INDEX_NAME)



# Function to search Spotify for song titles and artists
def search_spotify(query):
    results = sp.search(q=query, type='track', limit=10)  # Search for tracks, limit to top 10 results
    tracks = []
    for item in results['tracks']['items']:
        track_name = item['name']
        artist_name = item['artists'][0]['name']
        track_display = f"{track_name} by {artist_name}"
        tracks.append(track_display)
    return tracks




def similarity_search(search_input):
    """
    Search for similar songs based on the lyrics vector of the input song.
    
    Args:
        search_input (str): The song title and artist name.
    
    Returns:
        str: The most similar song ID if found, else None.
    """
    # Generate the vector for the input song's lyrics
    lyrics = get_lyrics_of_single_song(search_input)

    # I have to manually create query embedding here because it's easier

    res = openai.Embedding.create(input=lyrics[:8192], engine=MODEL)
    embedding = res['data'][0]['embedding']
    if not embedding:
        return None  # Cannot perform similarity search without a valid vector

    # Perform the similarity search using Pinecone
    result = index.query(vector=embedding, top_k=10)
    suggestion_tuples=[]
    for match in result['matches']:
        song_id = match['id']
        if fuzz.ratio(search_input.lower(), song_id.lower()) > 80:
            continue
        lyrics = get_lyrics_of_single_song(song_id)
        suggestion_tuples.append((song_id,lyrics))
        if len(suggestion_tuples) > 5:
            return suggestion_tuples  # Return the 5 most similar songs
    return None



def get_embed_url(search_input, sp):
    """
    Retrieve the Spotify embed URL for a given song title and artist name.

    Args:
        search_input (str): A string in the format "{song title} by {artist}".
        sp (spotipy.Spotify): An authenticated Spotipy client.

    Returns:
        str: The Spotify embed URL of the exact song if found, otherwise None.
    """
    # Split the search input into song title and artist
    if " by " in search_input:
        title, artist = search_input.split(" by ", 1)
    else:
        return None  # Invalid format, expected "TITLE by ARTIST"

    # Search for the track using the song title and artist name
    results = sp.search(q=f"track:{title} artist:{artist}", type='track', limit=5)  # Increase limit to get more options

    # Check for the most relevant result based on exact match
    for track in results['tracks']['items']:
        track_name = track['name'].lower()
        track_artists = ', '.join([a['name'].lower() for a in track['artists']])

        # Confirm the track name and artist match exactly
        if title.lower() == track_name and artist.lower() in track_artists:
            track_id = track['id']
            embed_url = f"https://open.spotify.com/embed/track/{track_id}"
            return embed_url
    
    # Return None if no exact match was found
    return None



def generate_song_description(input_lyrics, lyrics, input_song,suggestion_song):
    """
    Generate a witty, creative one-sentence description for given song lyrics using GPT-3.5.

    Args:
        lyrics (str): The song lyrics to describe.

    Returns:
        str: A witty, one-sentence description of the song lyrics.
    """
    # Construct the prompt for generating the description
    messages = [
        {"role": "system", "content": "You are a AI music reviewer that compares the lyrics of two songs and writes a 1 sentence description of why the songs are similar for a website's music catalog. Make sure your output ends with a period."},
        {"role": "user", "content": f"Create a 1 sentence description of why these songs' lyrics are similar. Use the input song name {input_song} and suggested song name {suggestion_song} in the sentence so it doesn't sound overly general. Use direct quotes from lyrics if useful:\n\n{input_lyrics}{lyrics}"}
    ]
    
    # Use GPT-3.5 turbo to generate the description
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=50,  # Allow for more tokens to generate a full description
        temperature=0.7,  # Increase temperature for more creativity
        n=1,
        stop=None
    )
    
    # Get the generated description
    description = response['choices'][0]['message']['content'].strip()
    
    return description







# Streamlit UI
st.title("Spotify Song Search")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ccffcc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Create three columns for side-by-side images
col1, col2, col3 = st.columns(3)

# Display an image in each column
with col1:
    st.image("Pics/spotifylogo.png", width=150)  # Adjust width as needed

with col2:
    st.image("Pics/openailogo.png", width=150)  # Adjust width as needed

with col3:
    st.image("Pics/pineconelogo.jpg", width=150)  # Adjust width as needed

# Text input box with suggestions (autocomplete functionality)
search_input = st.text_input("Type a song title or artist name:")

if search_input:
    # Fetch suggestions from Spotify API based on input
    suggestions = search_spotify(search_input)

    # Use columns to align the dropdown and the search button next to each other
    col1, col2 = st.columns([3, 1])  # Adjust column width ratio as needed

    # Display suggestions in a dynamic dropdown menu within column 1
    selected_song = col1.selectbox("Suggestions:", suggestions, key="dropdown")

    # Add a "Search" button in column 2
    search_button = col2.button("Search")

    # Only run similarity search if a song is selected and "Search" button is clicked
    if search_button and selected_song:
        st.write(f"You selected: {selected_song}")

        # Check if input song's lyrics are valid
        input_lyrics = get_lyrics_of_single_song(selected_song)

        # If lyrics are valid, run similarity search; otherwise, use vectorize the title only
        if input_lyrics:
            similar_songs = similarity_search(selected_song)
            if similar_songs:
                for tuple_item in similar_songs:
                    song=tuple_item[0]
                    lyrics=tuple_item[1]
                    st.write(f"Similar song found: **{song}**")
                    spotify_embed_url = get_embed_url(song,sp)
                    st.components.v1.iframe(spotify_embed_url, width=300, height=80)
                    description = generate_song_description(input_lyrics,lyrics,selected_song,song)
                    st.write(description)
                    st.write(f"**Lyrics**: {lyrics}")
            else:
                st.error("No similar songs found.")
        else:
            # Create embedding based off song title if MusixMatch can find the query song
            response = openai.Embedding.create(
            input=selected_song,
            engine=MODEL)  
            simple_rec = response['data'][0]['embedding']
            basic_sim_search = index.query(vector=simple_rec,top_k=1)
            st.write(basic_sim_search)          
