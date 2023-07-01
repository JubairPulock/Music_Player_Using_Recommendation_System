import streamlit as st
import pandas as pd
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

os.environ["SPOTIFY_CLIENT_ID"] = "7d72bb8569fe467eb05523bc6b0363d4"
os.environ["SPOTIFY_CLIENT_SECRET"] = "a4599d6166ef478f9650bf30a4fb7c5b"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                          client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))

data = pd.read_csv("data.csv")

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [int(results['year'])]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, data):
    try:
        song_data = data[data['name'] == song['name']].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, data)
        if song_data is None:
            st.warning(f"Warning: {song['name']} does not exist in Spotify or in the database")
            continue
        song_vector = song_data[number_cols].values
        if len(song_vector) != 15:
            continue
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict

def recommend_songs(song_list, data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, data)
    distances = cdist(np.atleast_2d(song_center), data[number_cols], 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

def main():

    st.set_page_config(layout='wide')
    st.image('https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GMikfjbGYyPGDTYdJuMwhg.png', width= 1380)

    song_name = st.text_input(f"Song Name")

    if st.button("Recommend"):
        st.subheader("Recommended Songs")
        song_list = [{'name': song_name, 'year': 0}]
        recommended_songs = recommend_songs(song_list, data)
        for song in recommended_songs:
            st.write(f"{song['name']} - {song['artists']} ({song['year']})")

if __name__ == "__main__":
    main()
