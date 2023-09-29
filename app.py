import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle 
import numpy as np 
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import re
from nltk.tokenize import WordPunctTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import itertools
import warnings
warnings.filterwarnings("ignore")

labels = ["pornografi", "sara", "radikalisme", "pencemaran nama baik"]

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
    st.image(wordcloud.to_array(), use_column_width=True, caption=f'WordCloud untuk label {text}.')

# Translate emoticon
emoticon_data_path = 'https://raw.githubusercontent.com/syahvan/indo-toxic-comment-classification/main/data/emoticon.txt'
emoticon_df = pd.read_csv(emoticon_data_path, sep='\t', header=None)
emoticon_dict = dict(zip(emoticon_df[0], emoticon_df[1]))

def translate_emoticon(t):
    for w, v in emoticon_dict.items():
        pattern = re.compile(re.escape(w))
        match = re.search(pattern,t)
        if match:
            t = re.sub(pattern,v,t)
    return t

def remove_newline(text):
    return re.sub('\n', ' ',text)

def remove_kaskus_formatting(text):
    text = re.sub('\[', ' [', text)
    text = re.sub('\]', '] ', text)
    text = re.sub('\[quote[^ ]*\].*?\[\/quote\]', ' ', text)
    text = re.sub('\[[^ ]*\]', ' ', text)
    text = re.sub('&quot;', ' ', text)
    return text

def remove_url(text):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)

def remove_excessive_whitespace(text):
    return re.sub('  +', ' ', text)

def tokenize_text(text, punct=False):
    text = WordPunctTokenizer().tokenize(text)
    text = [word for word in text if punct or word.isalnum()]
    text = ' '.join(text)
    text = text.strip()
    return text

slang_words = pd.read_csv('https://raw.githubusercontent.com/syahvan/indo-toxic-comment-classification/main/data/slangword.csv')
slang_dict = dict(zip(slang_words['original'],slang_words['translated']))

def transform_slang_words(text):
    word_list = text.split()
    word_list_len = len(word_list)
    transformed_word_list = []
    i = 0
    while i < word_list_len:
        if (i + 1) < word_list_len:
            two_words = ' '.join(word_list[i:i+2])
            if two_words in slang_dict:
                transformed_word_list.append(slang_dict[two_words])
                i += 2
                continue
        transformed_word_list.append(slang_dict.get(word_list[i], word_list[i]))
        i += 1
    return ' '.join(transformed_word_list)

def remove_non_alphabet(text):
    output = re.sub('[^a-zA-Z ]+', ' ', text)
    return output

def remove_twitter_ig_formatting(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'\brt\b', '', text)
    return text

def remove_repeating_characters(text):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))

stopword = pd.read_csv('https://raw.githubusercontent.com/syahvan/indo-toxic-comment-classification/main/data/stopwordbahasa.csv', header=None)
id_stopword_dict = stopword.rename(columns={0: 'stopword'})

def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text)
    text = text.strip()
    return text

# Buat objek stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    transformed_text = text.lower()
    transformed_text = remove_newline(text)
    transformed_text = remove_url(transformed_text)
    transformed_text = remove_twitter_ig_formatting(transformed_text)
    transformed_text = remove_kaskus_formatting(transformed_text)
    transformed_text = translate_emoticon(transformed_text)
    transformed_text = transformed_text.lower()
    transformed_text = remove_non_alphabet(transformed_text)
    transformed_text = remove_repeating_characters(transformed_text)
    transformed_text = remove_excessive_whitespace(transformed_text)
    transformed_text = tokenize_text(transformed_text)
    transformed_text = transform_slang_words(transformed_text)
    transformed_text = stemmer.stem(transformed_text)
    transformed_text = remove_stopword(transformed_text)
    transformed_text = transformed_text.lower().strip()
    return transformed_text


# Main
def main():
    st.title("Toxicity Classifier Dashboard")
    menu = st.sidebar.selectbox("Menu", ["Home", "Toxicity Classifier", "Wordcloud"])

    if menu == "Home":
        st.write("Selamat datang di dashboard Toxicity Classifier.")
    
    elif menu == "Toxicity Classifier":
        st.subheader("Prediksi Komentar Toxic Bahasa Indonesia")
        
        # Input Komentar
        input_text = st.text_area("Masukkan Komentar:")
        
        # Tombol prediksi
        if st.button("Prediksi"):
            # Lakukan prediksi
            tfidf = pickle.load(open("tf_idf.pkt", "rb"))
            text_tfidf = tfidf.transform([preprocess_text(input_text)])
            loaded_model = pickle.load(open('model_rf.pkt', "rb"))
            predicted_labels = loaded_model.predict(text_tfidf)
            probabilities = loaded_model.predict_proba(text_tfidf)
            predicted_labels = [labels[i] for i in range(len(labels)) if predicted_labels[0][i] == 1]
            proba_list = []
            for proba in probabilities:
                value = round(proba[0][1]*100, 2)
                proba_list.append(value)

            # Tampilkan label prediksi
            output_sentence = "Komentar tersebut toxic karena mengandung "
            if predicted_labels:
                if len(predicted_labels) == 1:
                    output_sentence += f"<span style='font-weight:bold;color:red'>{predicted_labels[0]}</span>"
                else:
                    labels_except_last = ", ".join(predicted_labels[:-1])
                    last_label = predicted_labels[-1]
                    output_sentence += f"<span style='font-weight:bold;color:red'>{labels_except_last}</span>"
                    output_sentence += f" dan <span style='font-weight:bold;color:red'>{last_label}</span>"
            else:
                output_sentence = "Komentar tersebut tidak toxic"

            st.markdown(output_sentence + '.', unsafe_allow_html=True)
            
            # Tampilkan grafik probabilitas
            prob_df = pd.DataFrame({'Label': labels, 'Probabilitas': proba_list})
            fig = px.bar(prob_df, x='Label', y='Probabilitas', color='Label', title='Grafik Probabilitas Komentar')
            fig.update_yaxes(range=[0, 100])
            hovertemp = "<b>Label: </b> %{x} <br>"
            hovertemp += "<b>Probabilitas: </b> %{y}<extra></extra>%"
            fig.update_traces(hovertemplate=hovertemp)
            st.plotly_chart(fig)
    
    elif menu == "Wordcloud":
        st.subheader("Wordcloud dari Dataset")

        # Pilihan label
        selected_label = st.selectbox("Pilih Label:", labels)
        
        # Tampilkan wordcloud
        generate_wordcloud(selected_label)

if __name__ == "__main__":
    main()

