import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import os
import spacy
import joblib

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
number_of_cores = joblib.cpu_count()
# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load data
news_path = "/Users/pedrocastro/Desktop/APCS/COMP 329 - NLP/Project/archive-4"
all_content = []


for csv_file in os.listdir(news_path):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(news_path, csv_file)
        df = pd.read_csv(csv_path, usecols=["content"])
        all_content.extend(df["content"].fillna("").astype(str).tolist())


# Preprocessing
stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)


def file_processing(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        token
        for token in tokens
        if token not in stop_words and token not in punctuation
    ]
    return " ".join(tokens)


preprocessed_content = joblib.Parallel(n_jobs=number_of_cores)(
    joblib.delayed(file_processing)(text) for text in all_content
)

# Vectorize using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_content)

# Fit LDA model
number_of_topics = 10
lda_model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
lda_model.fit(tfidf_matrix)

# Print top words for each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")


# Evaluate coherence
def calculate_topic_coherence(top_words, doc_tokens):
    coherence = 0
    for i in range(len(top_words)):
        for j in range(i + 1, len(top_words)):
            word1 = top_words[i]
            word2 = top_words[j]
            word1_vector = nlp(word1).vector
            word2_vector = nlp(word2).vector
            similarity = cosine_similarity([word1_vector], [word2_vector])[0][0]
            coherence += similarity
    return coherence / len(top_words)  # average coherence


topic_coherences = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    coherence = calculate_topic_coherence(top_words, preprocessed_content)
    topic_coherences.append(coherence)
    print(f"Topic {topic_idx} coherence: {coherence}")

# Calculate overall coherence
average_coherence = np.mean(topic_coherences)
print(f"Average coherence across all topics: {average_coherence}")

# time taken: 10 minutes

"""
Topic 0: like, one, said, says, people, time, first, new, game, show - Topic 0: General Life
Topic 1: company, percent, said, billion, million, market, companies, apple, year, new - Topic 1: Business and Finance
Topic 2: police, said, officers, told, man, authorities, according, county, officer, people - Topic 2: Law Enforcement and Crime
Topic 3: says, people, study, health, said, one, new, water, like, research - Topic 3: Health and Research
Topic 4: trump, clinton, campaign, republican, hillary, cruz, sanders, donald, said, presidential - Topic 4: Politics - Elections
Topic 5: trump, said, president, trade, china, european, eu, mr, united, minister - Topic 5: International Trade and Politics
Topic 6: trump, bill, republicans, obamacare, house, senate, health, tax, care, would - Topic 6: Politics - Legislation
Topic 7: court, students, law, school, said, rights, trump, women, people, state - Topic 7: Law and Education
Topic 8: said, syria, military, islamic, isis, iran, korea, syrian, forces, north - Topic 8: International Conflict and Military
Topic 9: trump, president, said, clinton, house, comey, russia, fbi, intelligence, russian - Topic 9: Politics - Government and Intelligence
Topic 0 coherence: 1.179916474223137
Topic 1 coherence: 1.4975828595459462
Topic 2 coherence: 1.4778383633121848
Topic 3 coherence: 1.3613626584410667
Topic 4 coherence: 1.8245962029322982
Topic 5 coherence: 1.971128974109888
Topic 6 coherence: 1.960355607792735
Topic 7 coherence: 1.925399836525321
Topic 8 coherence: 1.7242473922669888
Topic 9 coherence: 2.185788160003722
Average coherence across all topics: 1.7108216529153288 

"""
