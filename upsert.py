import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import openai
from fuzzywuzzy import fuzz
import streamlit as st
import re
from pinecone import Pinecone

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



MUSIXMATCH_API_KEY= st.secrets["MUSIXMATCH_API_KEY"]



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
    # Ensure the input string contains " by " to avoid unpacking errors
    if " by " in search_input:
        title, artist = search_input.split(" by ", 1)  # Split around the first occurrence of " by "
    else:
        return "Invalid input format. Expected 'TITLE by ARTIST'."

    # Proceed only if both title and artist are present
    if title and artist:
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
                lyrics = lyrics.split("*******")[0].strip()
                return lyrics
            else:
                return "Lyrics not found."
        else:
            return f"Error: {response.status_code} - Unable to retrieve lyrics."
    else:
        return "Invalid title or artist name."


# Main Controller Method for Upserting. Call this with a playlist_id to run batch upsert
# New method that loops through get_playlist_tracks and, and calls get_lyrics_of_single_song for each track, then upserts to Pinecone
def upsert_songs_individually(sp,playlist_id):
    tracks = get_playlist_tracks(sp,playlist_id)
    for track in tracks:
        if " by " in track:  # Ensure the format is correct
            title, artist = track.split(" by ", 1)  # Split around the first occurrence of " by "
            print(f"Title: {title}, Artist(s): {artist}")
            lyrics = get_lyrics_of_single_song(track)
            print(lyrics)
            vector = vectorize_single_song(lyrics,title,artist)
            index.upsert(vectors=[vector])
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
    

#upsert_songs_individually(sp,'37i9dQZF1EIW8xRaYy9ebd')