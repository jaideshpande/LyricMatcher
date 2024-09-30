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


# Function to classify whether the text is actual song lyrics or not using GPT-3.5
def is_lyrics(text):
    """
    Use GPT-3.5 to classify whether a given text is likely to be song lyrics.
    
    Args:
        text (str): The text to classify.
    
    Returns:
        bool: True if the text is classified as lyrics, False otherwise.
    """
    # Construct the prompt for classification
    messages = [
        {"role": "system", "content": "You are an AI language model that classifies text as song lyrics or not."},
        {"role": "user", "content": f"Classify the following text as 'Lyrics' or 'Non-Lyrics':\n\nText: {text}\n\nClassification:"}
    ]
    
    # Use GPT-3.5 turbo to classify the text
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=5,
        temperature=0.3,
        n=1,
        stop=["\n"]
    )
    
    # Get the classification result
    classification = response['choices'][0]['message']['content'].strip()
    
    return classification == "Lyrics"


def vectorize_lyrics(search_input):
    lyrics = retrieve_lyrics(search_input)  # Retrieve lyrics for the input song
    
    # Check if lyrics are retrieved and within the token length limit
    if lyrics and is_lyrics(lyrics):
        # Limit the lyrics to 8192 characters to ensure compatibility with the OpenAI model
        lyrics = lyrics[:8192]
        
        # Create the embedding using the OpenAI API
        res = openai.Embedding.create(
            input=lyrics,
            engine=MODEL
        )
        query_vector_embedding = res['data'][0]['embedding']
        return query_vector_embedding
    else:
        print(f"No valid lyrics found for {search_input}. Cannot create embedding.")        
        return None

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


# Function to classify whether the text is actual song lyrics or not using GPT-3.5
def is_lyrics(text):
    """
    Use GPT-3.5 to classify whether a given text is likely to be song lyrics.
    
    Args:
        text (str): The text to classify.
    
    Returns:
        bool: True if the text is classified as lyrics, False otherwise.
    """
    messages = [
        {"role": "system", "content": "You are an AI language model that classifies text as song lyrics or not."},
        {"role": "user", "content": f"Classify the following text as 'Lyrics' or 'Non-Lyrics':\n\nText: {text}\n\nClassification:"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=5,
        temperature=0.3,
        n=1,
        stop=["\n"]
    )
    
    classification = response['choices'][0]['message']['content'].strip()
    return classification == "Lyrics"

# Function to get Spotify recommendations if lyrics are not valid
def get_spotify_recommendation(track_name):
    """
    Get Spotify song recommendations based on the track name.
    
    Args:
        track_name (str): The name of the song to get recommendations for.
    
    Returns:
        str: Recommended song titles and artists.
    """
    results = sp.search(q=track_name, type='track', limit=3)
    recommendations = []

    for item in results['tracks']['items']:
        track_name = item['name']
        artist_name = item['artists'][0]['name']
        recommendations.append(f"{track_name} by {artist_name}")

    if recommendations:
        return f"Recommended songs:\n" + "\n".join(recommendations)
    else:
        return "No recommendations found."
    


def retrieve_lyrics(search_input):
    """
    Retrieve lyrics for a given search input and validate using GPT-3.5.

    Args:
        search_input (str): The song title and artist name.
    
    Returns:
        str: Valid lyrics if found, else None.
    """
    try:
        song = genius_object.search_song(title=search_input)
        if song and song.lyrics:
            lyrics = song.lyrics

            # Use GPT-3.5 to classify if the retrieved lyrics look like actual lyrics
            if is_lyrics(lyrics):
                return lyrics
            else:
                print("The retrieved text does not look like song lyrics. Falling back to Spotify recommendations...")
                return None  # Indicate invalid lyrics
        else:
            print(f"No lyrics found for {search_input}.")
            return None
    except Exception as e:
        print(f"Error retrieving lyrics for {search_input}: {e}")
        return None

def similarity_search(search_input):
    """
    Search for similar songs based on the lyrics vector of the input song.
    
    Args:
        search_input (str): The song title and artist name.
    
    Returns:
        str: The most similar song ID if found, else None.
    """
    # Generate the vector for the input song's lyrics
    vector = vectorize_lyrics(search_input)
    if not vector:
        return None  # Cannot perform similarity search without a valid vector

    # Perform the similarity search using Pinecone
    result = index.query(vector=vector, top_k=10)
    for match in result['matches']:
        song_id = match['id']
        if fuzz.ratio(search_input.lower(), song_id.lower()) > 80:
            continue
        lyrics = retrieve_lyrics(song_id)
        if lyrics:
            return song_id  # Return the first valid song ID
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
        lyrics = retrieve_lyrics(selected_song)

        # If lyrics are valid, run similarity search; otherwise, use Spotify recommendations
        if lyrics:
            similar_song = similarity_search(selected_song)
            if similar_song:
                st.write(retrieve_lyrics(selected_song))
                st.write(f"Similar song found: {similar_song}")
                #st.write(retrieve_lyrics(similar_song))
            else:
                st.error("No similar songs found.")
        else:
            # Fallback to Spotify recommendations if lyrics are not valid
            response = openai.Embedding.create(
            input=selected_song,
            engine=MODEL)  
            simple_rec = response['data'][0]['embedding']
            basic_sim_search = index.query(vector=simple_rec,top_k=1)
            st.write(basic_sim_search)          
