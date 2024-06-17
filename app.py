import re
import string
import nltk
import pickle
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Keep words with length > 1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

# loading the models here
tfidf = pickle.load(open(r'Models\\vectorizer.pkl', 'rb'))
model = pickle.load(open(r'Models\\model.pkl', 'rb'))

# creating the streamlit app here
st.title("Movie Genre Classifier App")
desc_input = st.text_area("Enter the Movie Synopsis below")

if st.button('Predict'):
    # cleaning the text here.
    cleaned_text = clean_text(desc_input)
    # vectorizing the cleaned text here
    vector_input = tfidf.transform([cleaned_text])
    # prediction result
    genre = model.predict(vector_input)[0]
    st.subheader(genre.title())