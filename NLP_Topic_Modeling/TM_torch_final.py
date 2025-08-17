import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nltk
import string
import spacy
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

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
stop_words = set(nltk.corpus.stopwords.words("english"))
punctuation = set(string.punctuation)


def file_processing(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    tokens = [
        token
        for token in tokens
        if token not in stop_words and token not in punctuation
    ]
    return " ".join(tokens)


preprocessed_content = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(file_processing)(text) for text in all_content
)


# Custom PyTorch Dataset
class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create dataset and dataloader
news_dataset = NewsDataset(preprocessed_content)
data_loader = DataLoader(news_dataset, batch_size=64, shuffle=True)

# Tokenization and vocabulary building
tokenizer = nltk.tokenize.word_tokenize
vocab = set()

for text in preprocessed_content:
    tokens = tokenizer(text)
    vocab.update(tokens)

vocab = {word: idx for idx, word in enumerate(vocab)}


# Vectorization using TF-IDF (Term Frequency-Inverse Document Frequency)
def tfidf_vectorizer(data):
    tokenized_data = [" ".join(tokenizer(text)) for text in data]
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_data)
    return tfidf_matrix


# Fit LDA model
number_of_topics = 10
lda_model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)

# Train LDA model
tfidf_matrix = tfidf_vectorizer(preprocessed_content)
lda_model.fit(tfidf_matrix)

# Print top words for each topic
feature_names = list(vocab.keys())
for topic_idx, topic in enumerate(lda_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Evaluate coherence
nlp = spacy.load("en_core_web_sm")


def calculate_topic_coherence(top_words, doc_tokens):
    coherence = 0
    for i in range(len(top_words)):
        for j in range(i + 1, len(top_words)):
            word1 = top_words[i]
            word2 = top_words[j]
            word1_vector = nlp(word1).vector
            word2_vector = nlp(word2).vector

            # Check if vectors are valid
            if len(word1_vector) == len(word2_vector) == 300 and len(word1_vector) > 0:
                similarity = F.cosine_similarity(
                    torch.tensor(word1_vector), torch.tensor(word2_vector)
                ).item()
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

# time spent:

"""
CODE OUTPUT: 23 minutes

Topic 0: zika, yankees, mosquito, virus, mosquitoes, microcephaly, aedes, girardi, aegypti, dengue - diseases 
Topic 1: rousseff, temer, brazil, mourinho, lula, petrobras, brazilian, dilma, guardiola, odebrecht - brazilian politics
Topic 2: rousey, kohlhepp, sandusky, paterno, ufc, lubitz, sager, rockettes, wildstein, demme - sports
Topic 3: cosby, advertisement, constand, spieth, cetin, cassini, pluto, juno, mickelson, rhoden - entertainment
Topic 4: doping, biles, wada, athletes, rodchenkov, efimova, antidoping, ioc, raisman, iaaf - sports doping
Topic 5: said, percent, health, company, federal, would, billion, new, tax, year - economy
Topic 6: said, police, one, people, like, says, new, time, two, first - crime
Topic 7: jammeh, dassey, marawi, gambia, sunedison, sponsors, updates, barrow, chapecoense, partners - sports 
Topic 8: trump, clinton, said, president, obama, campaign, would, donald, republican, house - politics
Topic 9: yahoo, shares, nasdaq, redstone, viacom, crude, billion, inc, highs, percent - economy & business

Topic 0 coherence: 0.0
Topic 1 coherence: 0.0
Topic 2 coherence: 0.0
Topic 3 coherence: 0.0
Topic 4 coherence: 0.0
Topic 5 coherence: 0.0
Topic 6 coherence: 0.0
Topic 7 coherence: 0.0
Topic 8 coherence: 0.0
Topic 9 coherence: 0.0
Average coherence across all topics: 0.0

x
"""
