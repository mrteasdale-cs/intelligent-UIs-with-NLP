import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import nltk
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from srsly.ruamel_yaml import BytesIO

nltk.download('stopwords')
import textblob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Sentiment Analysis Functions

def preprocessing(text):
    # convert to lowercase
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    tokens = word_tokenize(text)
    clean_tokens = []
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    for i, token in enumerate(tokens):
        if token not in stop_words:
            clean_tokens.append(token)

    lemmatizer = WordNetLemmatizer()
    for i, token in enumerate(clean_tokens):
        clean_tokens[i] = lemmatizer.lemmatize(token).lower().strip()

    return ' '.join(clean_tokens)

def analyse_sentiment(input_text):

    blob = TextBlob(input_text)

    # get sentiment using sentiment function
    sentiment = blob.sentiment
    print(sentiment)

    # get polarity and subjectivity
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    if polarity > 0:
        sentiment_lbl = "Positive"
    elif polarity < 0:
        sentiment_lbl = "Negative"
    else:
        sentiment_lbl = "Neutral"

    # Print the results
    result = {'sentiment': sentiment_lbl, 'polarity': polarity, 'subjectivity': subjectivity}

    return result

def preprocess_sentence_stopwords(text):
    # Tokenise and remove stopwords
    stop_words = set(stopwords.words('english'))
    sentence = nltk.word_tokenize(text)
    filtered_tokens = []
    for token in sentence:
        if token.lower() not in stop_words:
            filtered_tokens.append(token)

    cleaned_sentence = ' '.join(filtered_tokens)
    cleaned_sentence = re.sub(r'\s+([.,!?])', r'\1', cleaned_sentence)  # Attach punctuation

    return cleaned_sentence

def build_similarity_matrix(sentences):
    """Build a similarity matrix for the sentences using TF-IDF"""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix

def summarise_text(text, max_sentences=8):
    ranked_sentences = []
    summarised_text = ""
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = [preprocess_sentence_stopwords(sentence) for sentence in sentences]

    # Filter out empty sentences to avoid errors
    non_empty_sentences = [s for s in preprocessed_sentences if s.strip() != ""]
    if not non_empty_sentences:
        return "Summary unavailable: text contains only stopwords."

    similarity_matrix = build_similarity_matrix(non_empty_sentences)

    # generate a matrix of rankings
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    # sort and rank top sentences
    for i, s in enumerate(sentences):
        if preprocessed_sentences[i].strip() != "":
            ranked_sentences.append((scores[i], s.strip()))
    sorted_ranked_sentences = sorted(ranked_sentences, reverse=True)

    # join the ranked sentences to the summarised text, up to the max sentence provided parameter
    for ranked_sentence in sorted_ranked_sentences[:max_sentences]:
        summarised_text += " " + ranked_sentence[1]

    return summarised_text

def generate_wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        random_state=42
    ).generate(text)
    img = BytesIO()
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    #plt.show() NOT NEEDED
    plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    img.seek(0)
    return img