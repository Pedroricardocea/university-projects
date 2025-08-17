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

# Download NLTK resources if not already downloaded
import nltk

nltk.download("wordnet")

# Define preprocessing function
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    tokens = simple_preprocess(text, deacc=True)  # Tokenization
    tokens = [token for token in tokens if token not in STOPWORDS]  # Stopwords removal
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return tokens


# Define the path to the corpus
corpus_path = "/Users/pedrocastro/Desktop/APCS/COMP 329 - NLP/Project/archive-4"


# Define a corpus iterator
class MyCorpus:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for csv_file in os.listdir(self.path):
            if csv_file.endswith(".csv"):
                csv_path = os.path.join(self.path, csv_file)
                df = pd.read_csv(csv_path, usecols=["content"])
                for text in df["content"].fillna("").astype(str):
                    yield preprocess(text)


if __name__ == "__main__":
    freeze_support()
    # Create a dictionary

    print("Starting...")

    model_file = "lda_model.model"

    if os.path.exists(model_file):
        print("Loading LDA model...")
        lda_model = models.LdaModel.load(model_file)
        corpus = corpora.MmCorpus("corpus.mm")
        dictionary = corpora.Dictionary.load("corpus.dict")
        print("LDA model loaded.")
    else:
        dictionary = corpora.Dictionary(MyCorpus(corpus_path))

        print("Dictionary created.")

        # Save the dictionary
        dictionary.save("corpus.dict")
        print("Dictionary saved.")

        # Create a bag-of-words representation of the corpus
        corpora.MmCorpus.serialize(
            "corpus.mm", (dictionary.doc2bow(text) for text in MyCorpus(corpus_path))
        )
        print("Corpus serialized.")
        # Load the corpus and dictionary
        corpus = corpora.MmCorpus("corpus.mm")
        dictionary = corpora.Dictionary.load("corpus.dict")
        print("Corpus and dictionary loaded.")

        # Train Latent Semantic Indexing with 200D vectors
        lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=10)
        lda_model.save(model_file)
    print("LDA model trained.")

    # Print top 10 topics
    topics = lda_model.show_topics(num_topics=10, num_words=10)
    for topic_id, topic in topics:
        print(f"Topic {topic_id}: {topic}")

    # Evaluate coherence
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=MyCorpus(corpus_path),
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence = coherence_model.get_coherence()
    print(f"Coherence: {coherence}")


# time taken: 35 minutes

"""
Topic 0: 0.011*"said" + 0.010*"country" + 0.007*"president" + 0.007*"world" + 0.006*"state" + 0.006*"government" + 0.006*"united" + 0.005*"china" + 0.005*"russia" + 0.005*"minister" - Topic 0: International Relations
Topic 1: 0.008*"company" + 0.008*"people" + 0.006*"medium" + 0.006*"like" + 0.006*"news" + 0.005*"new" + 0.005*"time" + 0.005*"facebook" + 0.004*"said" + 0.004*"way" - Topic 1: Media Influence
Topic 2: 0.015*"game" + 0.013*"said" + 0.011*"team" + 0.009*"season" + 0.009*"year" + 0.009*"player" + 0.006*"time" + 0.006*"play" + 0.005*"win" + 0.005*"league" - Topic 2: Sports
Topic 3: 0.010*"percent" + 0.009*"year" + 0.009*"state" + 0.008*"said" + 0.007*"tax" + 0.006*"new" + 0.006*"million" + 0.005*"law" + 0.005*"government" + 0.005*"american" - Topic 3: Economy and Policy
Topic 4: 0.013*"said" + 0.008*"new" + 0.008*"city" + 0.007*"year" + 0.006*"water" + 0.004*"home" + 0.004*"building" + 0.004*"area" + 0.004*"car" + 0.004*"say" - Topic 4: Urban Development
Topic 5: 0.032*"said" + 0.013*"police" + 0.007*"told" + 0.007*"court" + 0.006*"officer" + 0.006*"case" + 0.006*"law" + 0.005*"city" + 0.005*"gun" + 0.005*"people" - Topic 5: Law and Crime
Topic 6: 0.046*"trump" + 0.013*"clinton" + 0.012*"said" + 0.012*"president" + 0.010*"republican" + 0.009*"campaign" + 0.007*"donald" + 0.007*"election" + 0.007*"obama" + 0.006*"house" - Topic 6: Political Campaigns
Topic 7: 0.011*"health" + 0.009*"study" + 0.009*"people" + 0.008*"woman" + 0.007*"food" + 0.007*"said" + 0.007*"drug" + 0.007*"child" + 0.006*"say" + 0.006*"doctor" - Topic 7: Health and Medicine
Topic 8: 0.017*"said" + 0.014*"attack" + 0.009*"muslim" + 0.009*"military" + 0.008*"syria" + 0.008*"state" + 0.008*"isi" + 0.008*"group" + 0.007*"force" + 0.007*"war" - Topic 8: Military and Conflict
Topic 9: 0.009*"like" + 0.007*"people" + 0.006*"woman" + 0.006*"time" + 0.006*"year" + 0.005*"life" + 0.005*"say" + 0.004*"thing" + 0.004*"way" + 0.004*"know" - Topic 9: General Life and Society:

0.4803692878573642

"""
