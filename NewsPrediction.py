import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.ensemble import RandomForestClassifier
import scipy
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

stopword = stopwords.words('english')
df = pd.read_csv("BBC News Train.csv")
le = LabelEncoder()
df['category_id'] = le.fit_transform(df['Category'])

def convert_lower(text):
   return text.lower()
 
def remove_tags(text):
  remove = re.compile(r'<.*?>')
  return re.sub(remove, '', text)

def special_char(text):
  reviews = ''
  for x in text:
    if x.isalnum():
      reviews = reviews + x
    else:
      reviews = reviews + ' '
  return reviews

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = word_tokenize(text)
  return [x for x in words if x not in stop_words]

def lemmatize_word(text):
  wordnet = WordNetLemmatizer()
  return " ".join([wordnet.lemmatize(word) for word in text])

# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# Train Model
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df.Text).toarray()
y = np.array(df.category_id.values)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, shuffle = True)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# Add css to make text bigger

st.markdown("<h1 style='text-align: center'>News Category Classifier</h1>", unsafe_allow_html=True)
st.text("\n\n")

if st.checkbox('Teams'):
  st.markdown("<h5 style=' color: white;'>2602125363 - Bernard Owens Wiladjaja </h1>", unsafe_allow_html=True)
  st.markdown("<h5 style=' color: white;'>2602172845 - Kevin Matthew Lunardi </h1>", unsafe_allow_html=True)
  st.markdown("<h5 style=' color: white;'>2602167196 - Nicolas Satria Dermawan </h1>", unsafe_allow_html=True)
  st.markdown("<h5 style=' color: white;'>2602172624 - Andrian Loria </h1>", unsafe_allow_html=True)
  st.markdown("<h5 style=' color: white;'>2602159723 - Deven Utama </h1>", unsafe_allow_html=True)
st.markdown("<h5 style=' color: white;'>Please Enter Your News Title Below</h1>", unsafe_allow_html=True)
input_data = st.text_area('')

if st.button('Predict'):
    # preprocessing
    # transform_data = transform_text(input_data)
    transform_data = convert_lower(input_data)
    transform_data = remove_tags(input_data)
    transform_data = special_char(input_data)
    transform_data = remove_stopwords(input_data)
    transform_data = lemmatize_word(input_data)
    
    # vectorize
    vector_data = vectorizer.transform([transform_data])

    # predict
    result = rfc.predict(vector_data)

    # display
    if result == 1:
        st.header(":red[Entertainment]")
    elif result == 2:
        st.header(":red[Politics]")
    elif result == 3:
        st.header(":red[Sport]")
    elif result == 4:
        st.header(":red[Tech]")
    else:
        st.header(":red[Business]")

