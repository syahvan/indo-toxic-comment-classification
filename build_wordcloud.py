import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Data training
data_train = pd.read_csv('https://raw.githubusercontent.com/syahvan/indo-toxic-comment-classification/main/data/train_processed.csv')
data_train.rename(columns={'pencemaran_nama_baik': 'pencemaran nama baik'}, inplace=True)

# Fungsi untuk menghasilkan wordcloud dari teks
def generate_wordcloud(text):
    # Filter teks yang sesuai dengan label
    label_text = " ".join(data_train[data_train[text] == 1]['processed_text'])

    # Membuat WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(label_text)

    # Menampilkan WordCloud dengan Streamlit
    st.image(wordcloud.to_array(), use_column_width=True, caption=f'WordCloud untuk Label: {text}')