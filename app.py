import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import Counter
import time
import lyricsgenius as lg
import openai
import os
from pinecone import Pinecone, ServerlessSpec
from fuzzywuzzy import fuzz
import streamlit as st
import langdetect
from langdetect import detect




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
genius_access_token ='-dg-PYd4zGePxuy4hBt-yquyOaNpySTTQOmxZc1ZPxvJkloVQqv8a1NFtidPFdBC'
genius_object=lg.Genius(genius_access_token)


#CONVERT TO ENV VARIABLES
openai.api_key="sk-proj-RtY23zzqdreL9a9g8eKTT3BlbkFJM10G3cjAs0GDnSKeJjSB"
openai.organization = "org-S93k6GSg97eTTmukpjvpTt14"

pc = Pinecone(
    api_key="dbe7110d-adaf-4f75-ab9f-41e99fc9533f"
)
index=pc.Index('openai')


def retrieve_lyrics(search_input): # lyrics of song using Genius API
    song=genius_object.search_song(title=search_input)
    lyrics = song.lyrics
    return lyrics


def vectorize_lyrics(search_input):
    lyrics = retrieve_lyrics(search_input) # gets lyrics of currently playing track with retrieve_lyrics. load_query_vector updates the vector with unique id
    res = openai.Embedding.create(
    input = 
        lyrics
    , engine=MODEL)
    query_vector_embedding = res['data'][0]['embedding']
    return query_vector_embedding 


# Runs similarity search between currently playing song and all songs in the vector db. Returns song_and_artist string
def similarity_search(search_input):
    result = index.query(vector=vectorize_lyrics(search_input), top_k=10)
    print(result)
    
    # Iterate through the top matches and find the first valid one
    for match in result['matches']:
        song_id = match['id']
        
        if fuzz.ratio(search_input.lower(), song_id.lower()) > 80:
            continue
        # Fetch the lyrics using Genius API
        lyrics = retrieve_lyrics(song_id)
        
        # Check if the lyrics are too short or too long
        if len(lyrics) < 200 or len(lyrics) > 5000:  # Adjust thresholds as needed
            print(f"Lyrics for {song_id} are too short/long, moving to the next match.")
            index.delete(ids=song_id)
            continue

        # Detect language of the song title and lyrics
        try:
            title_lang = detect(song_id)   # Detect language of the song title
            lyrics_lang = detect(lyrics)   # Detect language of the lyrics
            
            # Check if the title and lyrics are in the same language
            if title_lang != lyrics_lang:
                print(f"Language mismatch for {song_id}: Title is {title_lang}, but lyrics are {lyrics_lang}. Skipping.")
                index.delete(ids=song_id)
                continue
        except langdetect.lang_detect_exception.LangDetectException:
            print(f"Could not detect language for {song_id} or lyrics, skipping.")
            continue
        
        # If no issues, return the song
        return song_id
    
    # Fallback in case no valid songs were found
    return result['matches'][0]['id']


# Streamlit App Interface
st.title("Music Similarity Search")

# Get input from the user
search_input = st.text_input("Enter a song title and artist (e.g., 'Wonderwall by Oasis'): ")

if st.button("Find Similar Songs"):
    if search_input:
        similar_song = similarity_search(search_input)
        if similar_song:
            st.write(f"Suggested song: {similar_song}")
            suggested_lyrics = retrieve_lyrics(similar_song)
            if suggested_lyrics:
                st.write(f"Lyrics: {suggested_lyrics[:1000]}...")  # Display the first 500 characters of the lyrics
        else:
            st.error("No similar songs found.")
    else:
        st.error("Please enter a valid song title and artist.")