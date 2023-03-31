import streamlit as st
# Basic Streamlit Settings
st.set_page_config(page_title='Music Recommendation', layout = 'wide', initial_sidebar_state = 'auto')
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from load_css import local_css
from PIL import Image
import pydeck as pdk
import plotly.figure_factory as ff
import base64
import streamlit.components.v1 as components
import webbrowser

# Loading css file
local_css("style.css")


# Sidebar Section
def spr_sidebar():
    with st.sidebar:
        # st.image(SPR_SPOTIFY_URL, width=60)
        st.info('**Music Recommendation**')
        home_button = st.button("About Us")
        rec_button = st.button('Recommendation Engine')
        
        
        st.success('By Batch-14')
        st.session_state.log_holder = st.empty()
        if home_button:
            st.session_state.app_mode = 'home'
        
        if rec_button:
            st.session_state.app_mode = 'recommend'
        






    


# Load Data and n_neighbors_uri_audio are helper functions inside Recommendation Page
# Loads the track from filtered_track_df.csv file
def load_data():
    df = pd.read_csv(
        "filtered_track_df.csv")
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df


genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

# Fetches the Nearest Song according to Genre start_year and end year.
def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"] == genre) & (
        exploded_track_df["release_year"] >= start_year) & (exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios


# Recommendation Page

def rec_page():
    
    st.header("RECOMMENDATION ENGINE")
   
    with st.container():
        col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col3:
        st.markdown("***Choose your genre:***")
        genre = st.radio(
            "",
            genre_names, index=genre_names.index("Pop"))
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider(
            'Select the year range',
            1990, 2019, (2015, 2019)
        )
        acousticness = st.slider(
            'Acousticness',
            0.0, 1.0, 0.5)
        danceability = st.slider(
            'Danceability',
            0.0, 1.0, 0.5)
        energy = st.slider(
            'Energy',
            0.0, 1.0, 0.5)
        instrumentalness = st.slider(
            'Instrumentalness',
            0.0, 1.0, 0.0)
        valence = st.slider(
            'Valence',
            0.0, 1.0, 0.45)
        tempo = st.slider(
            'Tempo',
            0.0, 244.0, 118.0)
        tracks_per_page = 12
        test_feat = [acousticness, danceability,
                     energy, instrumentalness, valence, tempo]
        uris, audios = n_neighbors_uri_audio(
            genre, start_year, end_year, test_feat)
        tracks = []
        for uri in uris:
            track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
                uri)
            tracks.append(track)
    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [
            genre, start_year, end_year] + test_feat
    current_inputs = [genre, start_year, end_year] + test_feat

    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page

    current_tracks = tracks[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    if st.session_state['start_track_i'] < len(tracks):
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            if i % 2 == 0:
                with col1:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

            else:
                with col3:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

    else:
        st.write("No songs left to recommend")

    st.code("Algorithms that I have used in this filtering are the k-nearest neighbours and Random Forest")       

    

    

# Home Page

def home_page():
    st.subheader('About Me')
    
    
    col1, col2 = st.columns(2)


    


# Conclusion Page

def conclusions_page():

    st.header('Conclusion ')
    st.subheader("Model Performance Summary")
    st.success("Accuracy Test Results")
    algo_accuracy = Image.open(
        'images/algos_accuracy.png')
    st.image(algo_accuracy, width=400)
    st.write('Using a dataset of 228, 000 Spotify Tracks, I was able to predict popularity(greater than 57 popularity) using audio-based metrics such as key, mode, and danceability without external metrics such as artist name, genre, and release date. The Random Forest Classifier was the best performing algorithm with 92.0 % accuracy and 86.4 % AUC. The Decision Tree Classifier was the second best performing algorithm with 87.5 % accuracy and 85.8 % AUC.')

    st.write('Moving forward, I will use a larger Spotify database by using the Spotify API to collect my own data, and explore different algorithms to predict popularity score rather than doing binary classification.')

    algo_auc = Image.open(
        'images/models_auc_area_under_curve.png')
    st.image(algo_auc, width=400)
    st.write("The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classess. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.")
     
    st.subheader("Music Trend Analysis")
    '''
    - Dataset is imbalanced with more Europian countries.
    - Few songs have managed to make in 96% of Top Charts of all countries.
    - Even though few artists have many occurances in the Top charts, they don't have any song in Top 10 tracks occurances.
    - Average song duration preferred by most is around 3:00 minutes to 3:20 minutes
    - People in Asian countries prefer longer song duration. Europian countries mostly listen to songs close to or less than 3 minutes.
    - Asian countries prefer less of explicit songs, only 18%, compared to world average of 35%. "Global Top 50 Chart" has 23 explicit songs.
    '''
    



st.session_state.app_mode = 'recommend'

def main():
    
    spr_sidebar()
    st.header("Music Recommendation ")
    st.markdown(
        '*Music recommendation* is a online Robust Music Recommendation Engine where in you can finds the best songs that suits your taste.  ') 
    
    

    if st.session_state.app_mode == 'recommend':
        rec_page()

 

    if st.session_state.app_mode == 'conclusions':
        conclusions_page()

    if st.session_state.app_mode == 'home':
        home_page()



# Run main()
if __name__ == '__main__':
    main()


