from preprocessing import preprocess_text
from build_wordcloud import generate_wordcloud
import streamlit as st
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
import warnings
warnings.filterwarnings("ignore")

labels = ["pornografi", "sara", "radikalisme", "pencemaran nama baik"]

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
        
        # Pilih model
        model_name = st.selectbox("Pilih model:", ["Naive Bayes", "Random Forest", "XGBoost"])
        
        # Tombol prediksi
        if st.button("Prediksi"):
            # Lakukan prediksi
            tfidf = pickle.load(open("tf_idf.pkt", "rb"))
            text_tfidf = tfidf.transform([preprocess_text(input_text)])
            model_filename = f"model {model_name.lower()}.pkt"
            loaded_model = pickle.load(open(model_filename, "rb"))
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
                    output_sentence += predicted_labels[0]
                else:
                    output_sentence += ", ".join(predicted_labels[:-1])
                    output_sentence += " dan " + predicted_labels[-1]
            else:
                output_sentence = "Komentar tersebut tidak toxic"

            st.info(output_sentence + '.')
            
            
            # Tampilkan grafik probabilitas
            prob_df = pd.DataFrame({'Label': labels, 'Probabilitas': proba_list})
            fig = px.bar(prob_df, x='Label', y='Probabilitas', color='Label')
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


