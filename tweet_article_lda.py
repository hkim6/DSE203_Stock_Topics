# import nltk; nltk.download('stopwords')
import nltk

import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim import models

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
nlp = spacy.load("en_core_web_lg")
nlp.max_length = 1500000
import py_stringmatching as sm


class Topic_Classification:
    # Initialize Class, if topic_file = 0, utilize the article topic master_set, if it is 1 utilize the one for tweets
    def __init__(self, topic_file=0):
        if topic_file == 0:
            use_file = 'masterlists/articleTopics.csv'
        else:
            use_file = "masterlists/tweettopics.csv"
        self.topic_file = topic_file
        topic_df = pd.read_csv(use_file)
        master_list = []
        for topic in set(topic_df['topic']):
            master_list.append((topic, topic_df.loc[topic_df['topic'] == topic].reset_index(drop=True)))
        self.master_list = master_list

    # Perform LDA on the tweets
    def tweet_lda(self, df_tweets):
        # Drop tweets that are identical to ensure no saturation for classification
        al_tok = sm.AlphabeticTokenizer()
        jac = sm.Jaccard()
        drop_list = []
        for i, text in enumerate(df_tweets['content']):
            tw_tok = al_tok.tokenize(text.lower())
            for j in range(i + 1, len(df_tweets['content'])):
                tw2_tok = al_tok.tokenize(df_tweets['content'][j].lower())
                if jac.get_raw_score(tw_tok, tw2_tok) > .9:
                    drop_list.append(j)
        df_tweets = df_tweets.drop(drop_list).reset_index(drop=True)

        tweetDocuments = []  # List to store all documents in the training corpus as a 'list of lists'
        tweetWordCloud = []  # Single list version of the training corpus documents for WordCloud

        # Loop through all documents in the training corpus and combine into single list
        for tweet in df_tweets['content']:

            # Clean text removing escape characters and spaces
            tweet = tweet.strip()
            tweet = tweet.replace('\n', ' ').replace('\r', '').replace(' ', ' ').replace(' ', ' ')
            while '  ' in tweet:
                tweet = tweet.replace('  ', ' ')

            # Parse document with SpaCy
            tweetdoc = nlp(tweet)
            tweetlist = []

            # Keep relevant words and apply lemmatization
            for token in tweetdoc:
                # Remove strings that contain 'http' and '@' which commonly appear in tweets but do not help context
                if 'http' not in str(token) and '@' not in str(token):
                    if token.is_stop == False and token.is_punct == False and (
                            token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "VERB"):
                        tweetlist.append(token.lemma_.lower())
                        tweetWordCloud.append(token.lemma_.lower())
            # Add to list for training corpus
            tweetDocuments.append(tweetlist)

        # Create bag of words and fit TF-IDF model
        ID2word = corpora.Dictionary(tweetDocuments)

        corpus = [ID2word.doc2bow(doc) for doc in tweetDocuments]
        TFIDF = models.TfidfModel(corpus)
        trans_TFIDF = TFIDF[corpus]

        # Train LDA model
        lda_model = gensim.models.LdaMulticore(corpus=trans_TFIDF, num_topics=3, id2word=ID2word,
                                               random_state=42, alpha=0.9, eta=0.35, passes=100)

        # Write topics to list with 20 words each
        t_list = lda_model.print_topics(num_words=20)
        # Call Topic_Classification method to get back the topic label
        ret_topic = Topic_Classification.assign_topic(t_list, self.master_list)

        return ret_topic

    def article_lda(self, df_articles):
        al_tok = sm.AlphabeticTokenizer()
        jac = sm.Jaccard()
        drop_list = []
        for i, text in enumerate(df_articles['content']):
            tw_tok = al_tok.tokenize(text.lower())
            for j in range(i + 1, len(df_articles['content'])):
                tw2_tok = al_tok.tokenize(df_articles['content'][j].lower())
                if jac.get_raw_score(tw_tok, tw2_tok) > .9:
                    drop_list.append(j)
        df_articles = df_articles.drop(drop_list).reset_index(drop=True)

        articleDocuments = []  # List to store all documents in the training corpus as a 'list of lists'
        articleWordCloud = []  # Single list version of the training corpus documents for WordCloud

        # Loop through all documents in the training corpus and combine into single list
        for article in df_articles['content']:

            # Clean text removing escape characters and spaces
            article = article.strip()
            article = article.replace('\n', ' ').replace('\r', '').replace(' ', ' ').replace(' ', ' ')
            while '  ' in article:
                article = article.replace('  ', ' ')  # Remove extra spaces

            for num in range(0, 10):
                article = article.replace(str(num), '')

            # Parse document with SpaCy
            artdoc = nlp(article)
            artlist = []

            # Keep relevant words and apply lemmatization
            for token in artdoc:
                if token.is_stop == False and token.is_punct == False and (
                        token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "VERB"):
                    artlist.append(token.lemma_.lower())
                    articleWordCloud.append(token.lemma_.lower())
            # Add to list for training corpus
            articleDocuments.append(artlist)

        # Create bag of words and fit TF-IDF model
        ID2word = corpora.Dictionary(articleDocuments)

        corpus = [ID2word.doc2bow(doc) for doc in articleDocuments]
        TFIDF = models.TfidfModel(corpus)
        trans_TFIDF = TFIDF[corpus]

        # Train LDA model
        lda_model = gensim.models.LdaMulticore(corpus=trans_TFIDF, num_topics=3, id2word=ID2word,
                                               random_state=42, alpha=0.9, eta=0.35, passes=100)
        # Write topics to list with 20 words each
        t_list = lda_model.print_topics(num_words=20)
        # Call Topic_Classification method to get back the topic label
        ret_topic = Topic_Classification.assign_topic(t_list, self.master_list)

        return ret_topic


    # Perform classification on the returned topics of the grouped tweets and return a topic name.
    def assign_topic(topic_bows, master_list):
        max_value = 0
        # Loop through the topic Bag of Words and weights to gain scores for each
        for bow in topic_bows:
            for i in range(len(master_list)):
                score = Topic_Classification.compare_tuple_to_topic(bow, master_list[i][1])
                # If the score is greater than the previous stored max value, then store it as new max value save topic
                if score > max_value:
                    max_value = score
                    top_topic = master_list[i][0]
        # If max value is less tha 0.02, assign it as miscellaneous category since score it too low
        if max_value < 0.02:
            return "misc"
        else:
            return top_topic

    # Compare bag of words and weights of unlabelled tweets against the weights of master list and return score
    @staticmethod
    def compare_tuple_to_topic(tuple1, df1):
        df2 = Topic_Classification.tuple_to_df(tuple1)
        df = df1.set_index('key').join(df2.set_index('key'), on='key', lsuffix='1', rsuffix='2', how='inner')
        df['weight'] = df['weight1'] + df['weight2']
        return df['weight'].sum() / len(tuple1)

    # Convert tuple to dataframe
    @staticmethod
    def tuple_to_df(tuple1):
        step1 = tuple1[1].split(" + ")
        step2 = []
        for i in range(len(step1)):
            step2.append(step1[i].split("*"))

        df = pd.DataFrame(step2, columns=['weight', 'key'])
        df['count'] = 1
        df = df.astype({'weight': 'float64'})

        return df