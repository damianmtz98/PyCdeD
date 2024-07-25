import streamlit as st
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter
from nltk.corpus import stopwords
import nltk
import re

# descargamos las stopwords de nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(set(stopwords.words('spanish')))

# cargamos el modelo de spacy para español
nlp = spacy.load("es_core_news_sm")
analyzer = SentimentIntensityAnalyzer()

# función para analizar el sentimiento en inglés
def analyze_sentiment_english(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] > 0:
        return 'POSITIVE'
    elif sentiment['compound'] < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

# función para analizar el sentimiento en español
def analyze_sentiment_spanish(text):
    doc = nlp(text)
    sentiment = analyzer.polarity_scores(" ".join([token.lemma_ for token in doc]))
    if sentiment['compound'] > 0:
        return 'POSITIVE'
    elif sentiment['compound'] < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

# función para analizar el sentimiento según el idioma
def analyze_sentiment(text, language):
    if language == 'English':
        return analyze_sentiment_english(text)
    elif language == 'Spanish':
        return analyze_sentiment_spanish(text)
    else:
        return 'UNKNOWN'

# limpiamos el texto
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^A-Za-z\s]', '', text)  # eliminamos caracteres no alfabéticos
    return text.lower()

# contamos palabras, excluyendo stopwords
def count_words(text_series):
    all_words = ' '.join(map(clean_text, text_series)).split()
    filtered_words = [word for word in all_words if word not in stop_words and word.isalpha()]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(10)

# título de la app
st.title('Análisis de Sentimientos de Reseñas y Publicaciones')

# información sobre los idiomas soportados e limitaciones de la app
st.write("""
### Parámetros de la aplicación:
- **Idiomas soportados**: Inglés, Español.
- **Métodos de entrada**: Archivo CSV, Texto manual.
- **Resultados**: Análisis de sentimientos en texto y números.
""")

# seleccionamos el idioma
language = st.selectbox("Selecciona el idioma del texto:", ["English", "Spanish"])

# función para subir archivo de texto o ingresar manualmente
input_method = st.radio("Elige el método de entrada", ["Subir archivo CSV", "Ingresar texto manualmente"])

if input_method == "Subir archivo CSV":
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
    if uploaded_file is not None:
        try:
            reviews_df = pd.read_csv(uploaded_file, encoding='latin1')  # cambiamos la codificación según sea necesario
            st.markdown("<h3 style='text-align: left; color: white;'>Selecciona la columna que contiene las reseñas:</h3>", unsafe_allow_html=True)
            review_column = st.selectbox("", reviews_df.columns)
            # analizamos los sentimientos
            reviews_df['sentiment'] = reviews_df[review_column].apply(lambda x: analyze_sentiment(str(x), language))
            st.write(reviews_df.head())

            # mostramos números
            sentiment_counts = reviews_df['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']
            total_reviews = sentiment_counts['count'].sum()
            st.write(f"Recuento de sentimientos (Total: {total_reviews}):")
            st.write(sentiment_counts)

            # gráfico de distribución de sentimientos
            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x='sentiment',
                y='count',
                color='sentiment'
            )
            st.altair_chart(chart, use_container_width=True)

            # filtramos reseñas por sentimiento
            sentiment_filter = st.selectbox("Filtrar reseñas por sentimiento", options=reviews_df['sentiment'].unique())
            filtered_reviews = reviews_df[reviews_df['sentiment'] == sentiment_filter]
            st.write(filtered_reviews)

            # estadísticas de palabras más comunes
            st.subheader("Palabras más utilizadas en las reseñas")
            common_words = count_words(reviews_df[review_column])
            st.write(pd.DataFrame(common_words, columns=["Palabra", "Frecuencia"]))

        except UnicodeDecodeError:
            st.error("Error al decodificar el archivo CSV. Por favor, verifica la codificación del archivo.")
elif input_method == "Ingresar texto manualmente":
    user_text = st.text_area("Ingresa las reseñas/publicaciones, separadas por un punto y coma (;)")
    if user_text:
        reviews = user_text.split(";")
        reviews_df = pd.DataFrame(reviews, columns=['review'])
        reviews_df['sentiment'] = reviews_df['review'].apply(lambda x: analyze_sentiment(str(x), language))
        st.write(reviews_df.head())

        # mostramos números
        sentiment_counts = reviews_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        total_reviews = sentiment_counts['count'].sum()
        st.write(f"Recuento de sentimientos (Total: {total_reviews}):")
        st.write(sentiment_counts)

        # gráfico de distribución de sentimientos
        chart = alt.Chart(sentiment_counts).mark_bar().encode(
            x='sentiment',
            y='count',
            color='sentiment'
        )
        st.altair_chart(chart, use_container_width=True)

        # filtramos reseñas por sentimiento
        sentiment_filter = st.selectbox("Filtrar reseñas por sentimiento", options=reviews_df['sentiment'].unique())
        filtered_reviews = reviews_df[reviews_df['sentiment'] == sentiment_filter]
        st.write(filtered_reviews)

        # estadísticas de palabras más comunes
        st.subheader("Palabras más utilizadas en las reseñas")
        common_words = count_words(reviews_df['review'])
        st.write(pd.DataFrame(common_words, columns=["Palabra", "Frecuencia"]))

# input del usuario para nueva reseña
if input_method != "Subir archivo CSV":
    user_review = st.text_input("Ingresa una reseña")
    if user_review and st.button("Analizar Reseña"):
        sentiment = analyze_sentiment(user_review, language)
        st.write(f"Sentimiento: {sentiment}")
