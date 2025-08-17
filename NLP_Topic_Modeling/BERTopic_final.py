import os
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
import multiprocessing
from multiprocessing import freeze_support
from bertopic import BERTopic
from tqdm import tqdm

news_path = "/Users/pedrocastro/Desktop/APCS/COMP 329 - NLP/Project/archive-4"

all_content = []

for csv_file in os.listdir(news_path):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(news_path, csv_file)
        df = pd.read_csv(csv_path, usecols=["content"])
        all_content.extend(df["content"].fillna("").astype(str).tolist())

print(len(all_content))


print(len(all_content))
print(len(all_content[:50000]))

all_content = all_content[:50000]


model_path = "bertopic_model"

if os.path.exists(model_path):
    topic_model = BERTopic.load(model_path)
    print("Model loaded.")
else:
    topic_model = BERTopic(
        embedding_model="paraphrase-MiniLM-L6-v2", calculate_probabilities=True
    )
    topic_model.fit(all_content)
    topic_model.save(model_path)
    print("Model created.")


topics, probs = topic_model.fit_transform(all_content)


print(topic_model.get_topic_info())

print(topic_model.get_topic(0))

topic_model.visualize_topics()

print(topic_model.get_representative_docs(0))
