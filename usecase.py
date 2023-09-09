
import pandas as pd
import json
import nltk
import re
import string
import pickle
import joblib
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

nltk.download('stopwords')
def clean(text): 
  

    # Special characters
    
    text=re.sub(r'^\d+$', "", text)
    text=re.sub(r'\d+', '',text)
    
    
    # Contractions
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"youve", "you have", text)  
    text = re.sub(r"donå«t", "do not", text)   
    # usernames mentions like "@abc123"        
    ment = re.compile(r"(@[A-Za-z0-9]+)")
    text =  ment.sub(r'', text)
    # Character entity references
    text = re.sub(r"&amp;", "&", text)
    text =re.sub(r'^\d+$','',text)
    # html tags
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(html, '', text)
    
    # Urls
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)
    text = re.sub(r'https?://\S+|www\.\S+','', text)
        
    #Punctuations and special characters
    
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    
    text = text.lower()
    
    splits = text.split()
    splits = [word for word in splits if word not in set(nltk.corpus.stopwords.words('english'))]
    text = ' '.join(splits)
    
    
    return text




def remove_stopword(x):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(x)  # Tokenize the description into words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)  


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)  # Tokenize the text into words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text


with open('lg_model.pkl', 'rb') as model_file, \
    open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    lg_model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)
invalid_defects = pd.read_excel("cancelled_defects.xlsx")
st.title("Defect Management")
text_input = st.text_input("Enter a defect:")

if st.button("Classify Defect"):
    cleaned_text = clean(text_input)
    processed_text = remove_stopword(cleaned_text)
    #processed_text_string = ' '.join(processed_text)
    lemmatized_text = lemmatize_text(processed_text)
    tfidf_vector = vectorizer.transform([lemmatized_text])
    predicted_sentiment = lg_model.predict(tfidf_vector)[0]
    

    if predicted_sentiment == 0:
        st.write("The defect is classified as 'Invalid' ")
        tfidf_matrix_file = "tfidf_matrix_invalid_defects.pkl"
        try:
            tfidf_matrix = joblib.load(tfidf_matrix_file)

            cosine_similarities = linear_kernel(tfidf_vector, tfidf_matrix).flatten()

            similar_indices = cosine_similarities.argsort()[::-1]

            recommended_reasons = invalid_defects.iloc[similar_indices[:10]]['Description'].tolist()

            st.write("Top Reasons for 'Invalid' Defects:")
            for i, reason in enumerate(recommended_reasons, start=1):
                st.markdown(f"<p style='font-family: Arial; font-size: 14px;'>Defect Reason {i}: {reason}</p>", unsafe_allow_html=True)

        except FileNotFoundError:
            st.write("Error: TF-IDF matrix file not found.")
    else:
        st.write("The defect is classified as 'Valid.'")




