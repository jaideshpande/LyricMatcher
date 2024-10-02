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





MODEL = "text-embedding-ada-002"

#Spotify developer credentials. CONVERT TO ENV VARIABLES
client_id = '0fd41a617c7a41018be9a9cb8bcf2582'
client_secret = '7b977dcb49d64486ae0b1e9018562f45'
redirect_uri = 'http://localhost:3000/callback'
scope = 'playlist-read-collaborative playlist-read-private user-modify-playback-state user-read-playback-state'
# Set up the authentication
auth_manager = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope)
sp = spotipy.Spotify(auth_manager=auth_manager)

#CONVERT TO ENV VARIABLES
openai.api_key="sk-proj-RtY23zzqdreL9a9g8eKTT3BlbkFJM10G3cjAs0GDnSKeJjSB"
openai.organization = "org-S93k6GSg97eTTmukpjvpTt14"

pc = Pinecone(
    api_key="dbe7110d-adaf-4f75-ab9f-41e99fc9533f"
)
index=pc.Index('openai')



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



# Streamlit UI
st.title("Spotify Song Search")

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
        lyrics = get_lyrics_of_single_song(selected_song)

        # If lyrics are valid, run similarity search; otherwise, use vectorize the title only
        if lyrics:
            similar_songs = similarity_search(selected_song)
            if similar_songs:
                for tuple_item in similar_songs:
                    #st.write(get_lyrics_of_single_song(selected_song))
                    song=tuple_item[0]
                    lyrics=tuple_item[1]
                    st.write(f"Similar song found: **{song}**")
                    st.write(f"Lyrics: {lyrics}")
            else:
                st.error("No similar songs found.")
        else:
            # Create embedding based off song title
            response = openai.Embedding.create(
            input=selected_song,
            engine=MODEL)  
            simple_rec = response['data'][0]['embedding']
            basic_sim_search = index.query(vector=simple_rec,top_k=1)
            st.write(basic_sim_search)          
