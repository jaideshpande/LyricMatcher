import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import Counter
import time
import lyricsgenius as lg
import openai
import os
from pinecone import Pinecone, ServerlessSpec
from fuzzywuzzy import fuzz


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
    result = index.query(vector=vectorize_lyrics(search_input), top_k=3)
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
            continue
        
        # If no issues, return the song
        return song_id
    
    # Fallback in case no valid songs were found
    return result['matches'][0]['id']







# Code to Load VectorDB with songs from any playlist
def get_playlist_tracks(sp,playlist_id):
     # Ensure we have a token
    if not sp.auth_manager.get_access_token(as_dict=False):
        raise Exception("Failed to obtain access token")

    # Fetch tracks from the playlist
    results = sp.playlist_tracks(playlist_id)
    tracks = []
    while results:
        for item in results['items']:
            track = item['track']
            if track:  # Check if track details are present
                title = track['name']
                artists = ', '.join(artist['name'] for artist in track['artists'])
                tracks.append((title, artists))
        if results['next']:
            results = sp.next(results)
        else:
            break
    return tracks



def vectorize_playlist_songs_individually(playlist_id):
    tracklist = get_playlist_tracks(sp, playlist_id)

    for song in tracklist:
        try:
            # Get the song title and artist
            song_title = song[0]
            artist_name = song[1]
            search_title = f"{song_title} {artist_name}"

            # Fetch lyrics for one song at a time
            song_data = genius_object.search_song(title=search_title)
            if song_data is None:
                print(f"Lyrics not found for {song_title} by {artist_name}")
                continue
            lyrics = song_data.lyrics

            # Vectorize the song lyrics
            res = openai.Embedding.create(
                input=lyrics[:8192],  # stay within token limits
                engine=MODEL
            )
            embedding = res['data'][0]['embedding']
            unique_id = f"{song_title} by {artist_name}"
            ascii_id = ''.join([char for char in unique_id if char.isascii()])

            # Prepare the vector in the required format for Pinecone
            vector = {
                "id": ascii_id,
                "values": embedding
            }

            # Upsert the vector to Pinecone immediately
            index.upsert(vectors=[vector])
            print(f"Successfully upserted: {ascii_id}")
        
        except Exception as e:
            print(f"Error processing {song_title} by {artist_name}: {e}")
        finally:
            time.sleep(12)  # Delay to avoid rate limits from the Genius API

similar_lyrics = similarity_search("hey jude beatles")
print(similar_lyrics)
suggested_lyrics = retrieve_lyrics(similar_lyrics)
print(suggested_lyrics)
