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
import re
import requests




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
MUSIXMATCH_API_KEY='7f60bcdff18208710d478700e42cdd97'


#CONVERT TO ENV VARIABLES
openai.api_key="sk-proj-RtY23zzqdreL9a9g8eKTT3BlbkFJM10G3cjAs0GDnSKeJjSB"
openai.organization = "org-S93k6GSg97eTTmukpjvpTt14"

pc = Pinecone(
    api_key="dbe7110d-adaf-4f75-ab9f-41e99fc9533f"
)
index=pc.Index('openai')



def get_playlist_tracks(sp, playlist_id):
    """Retrieve tracks from a given Spotify playlist and return an array of 'TITLE by ARTIST' strings."""
    results = sp.playlist_tracks(playlist_id)
    tracks = []
    
    # Loop through all the results and append track details
    while results:
        for item in results['items']:
            track = item['track']
            if track:
                # Format each track as "TITLE by ARTIST"
                title = track['name']
                artists = ', '.join(artist['name'] for artist in track['artists'])
                track_info = f"{title} by {artists}"
                tracks.append(track_info)
        # Check if there are more pages to fetch
        if results['next']:
            results = sp.next(results)
        else:
            break
    
    return tracks


def get_lyrics_of_single_song(search_input):
    title, artist = search_input.split(" by ", 1)  # Split around the first occurrence of " by " to get title and artist
    """
    Get lyrics of a song using Musixmatch API.

    Args:
        title (str): Song title.
        artist (str): Artist name.
        api_key (str): Your Musixmatch API key.

    Returns:
        str: Lyrics of the song if found, else an error message.
    """
    # Set up the API endpoint and parameters
    url = "https://api.musixmatch.com/ws/1.1/matcher.lyrics.get"
    params = {
        "q_track": title,
        "q_artist": artist,
        "apikey": MUSIXMATCH_API_KEY
    }

    # Make a request to the Musixmatch API
    response = requests.get(url, params=params)
    
    # Check if the response is valid
    if response.status_code == 200:
        data = response.json()
        # Extract the lyrics from the response
        if data['message']['header']['status_code'] == 200:
            lyrics = data['message']['body']['lyrics']['lyrics_body']
            return lyrics
        else:
            return "Lyrics not found."
    else:
        return f"Error: {response.status_code} - Unable to retrieve lyrics."


# New method that loops through get_playlist_tracks and, and calls get_lyrics_of_single_song for each track, then upserts to Pinecone
def upsert_songs_individually(sp,playlist_id):
    tracks = get_playlist_tracks(sp,playlist_id)
    for track in tracks:
        if " by " in track:  # Ensure the format is correct
            title, artist = track.split(" by ", 1)  # Split around the first occurrence of " by "
            print(f"Title: {title}, Artist(s): {artist}")
            lyrics = get_lyrics_of_single_song(title,artist)
            print(lyrics)
            vector = vectorize_single_song(lyrics,title,artist)
            index.upsert(vectors=vector)
            print(f"Successfully upserted: {title} by {artist}")
            time.sleep(6)
        else:
            continue


def vectorize_single_song(lyrics,title,artist):
    res = openai.Embedding.create(input=lyrics[:8192], engine=MODEL)
    embedding = res['data'][0]['embedding']
    unique_id = f"{title} by {artist}"
    ascii_id = ''.join([char for char in unique_id if char.isascii()])

    # Prepare and upsert the vector to Pinecone
    vector = {"id": ascii_id, "values": embedding}
    return vector
    



