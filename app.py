import pandas as pd
import streamlit as st
import string
import pickle
import contractions
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english')) - set(['not'])

def cleanData(input):
 
  df_train = pd.DataFrame({'title+review' : [input], 'title+review_clean': ['']})

  df_train['title+review_clean'] = df_train['title+review'].str.lower()
  df_train['title+review_clean'] = df_train['title+review_clean'].str.replace('[^\w\s]', '', regex=True)
  df_train['title+review_clean'] = df_train['title+review_clean'].str.replace('\d+', '', regex=True)
  df_train['title+review_clean'] = df_train['title+review_clean'].apply(lambda x: ' '.join([contractions.fix(word) for word in str(x).split()]))
  df_train['title+review_clean'] = df_train['title+review_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopWords]))
  df_train['title+review_clean'] = df_train['title+review_clean'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word,pos='v') for word in x.split()]))

  return df_train['title+review_clean'][0]

st.title("Amazon Review Sentiment")
input = st.text_area("Enter your review", "This was a great purchase!")

input = cleanData(input)

#loading the word embedding for the new input
unpickled_tfidf = pickle.load(open('tfidf_vectorizer_fitted.pkl','rb'))
unpickled_tfidf.transform([input]).todense()

# Loading the best model 
best_model = pickle.load(open('best_model_fitted.pkl','rb'))

model_output = best_model.predict(unpickled_tfidf.transform([input]))

if st.button("Get Sentiment"):
    if model_output[0]==0:
        st.write('**Negative Review**')
    else:
        st.write('**Positive Review**')