# Overview

This repository contains a set of experiments for the [Incident Streams](http://www.trecis.org) track at [TREC 2019a](https://trec.nist.gov/).

We leverage machine learning models of increasing complexity for classifying Twitter content according to an ontology of 23 crisis responder information needs. 

# Methods

This work compares four machine learning-based methods for classifying tweets into various high-level information type, as specified by the 2018 TREC-IS information ontology.
The learning models evaluated in this experiment use a simple one-versus-rest (OVR) classification scheme and compare scenarios with/without semi-supervised expanded training data and with/without subword embeddings to capture context.
This section outlines model evaluations, model training strategies, and tweet featurization.

## Model Evaluation

While we evaluate combinations of learning scenarios, within each scenario, we evaluate two standard machine learning models as implemented in Python's [scikit-learn package](https://scikit-learn.org/): naive Bayes and random forests (RFs).
For each model, we run a randomized parameter search across 128 iterations using 10-fold cross-validation in which each model's performance is measured using macro F1 score.

For the naive Bayes implementation, we randomly sample $alpha$ and binarize parameters from uniform $(0,1)$ distributions and randomly select between fitting the class-label prior distribution.
For RFs, we sample parameters for maximum depth and per-tree feature count, minimum samples needed for splitting nodes and establishing leaves, and swap between Gini coefficient and entropy criterion.

Finally, we report F1 scores and accuracy for the final model in each of our four scenarios.

## Model Training

We train our models using the training and testing datasets from 2018 as well as the 2019a training dataset.

## Featurization

To convert tweets into feature vectors, we employ three methods: First, we extract a collection of Twitter-specific features to create a 15-dimensional numeric vector, including capitalization, whether the account is verified, whether the tweet is a retweet, the tweet's sentiment (using VADER \cite{Hutto2014}), as well as number of characters, hashtags, media links, mentions, and terms.

The second featurization method is a standard bag-of-words model.
We tokenize each tweet using NLTK's tweet tokenizer and apply a term-frequency, inverse-document-frequency (TF-IDF) weighting to the top 10,000 features.
To weight these features, we learn this TF-IDF feature weighting using a subsample of English tweets collected from Twitter's public sample stream between 2013 and 2016; we learn on this larger sample to increase generalizability in our data and address out-of-vocabulary issues in the original TREC-IS set.
In this vectorizer, we create unigrams and bigrams and remove rare terms (those that occur fewer than four times) and common terms (those that occur in more than half of the documents).

Our third vectorization method uses FastText subword embeddings to create a low-dimensional ($d=200$) embedding that captures the context around tokens.
We use a context window of 10 and require a term to appear at least five times in our data.
While FastText provides several pre-trained word vector datasets trained on Wikipedia and web pages, we build our own embedding model using a set of English tweets from Twitter's public sample stream, also from 2013-2016.

### Featurization Models

We provide the TF-IDF vectorizer built from a 1-percent sample of English tweets posted to Twitter and captured in Twitter's public sample stream between 2013 and 2016.
This dataset contains 11,715,393 tweets.
You can download this vectorizer here: [2013to2016_tfidf_vectorizer_20190109.pkl](http://obj.umiacs.umd.edu/trecis_2018/2013to2016_tfidf_vectorizer_20190109.pkl)

We also provide our FastText-trained model on this same set of English tweets, which you can find here: [archived_text_sample_2013to2016_gensim_200.model.tgz](http://obj.umiacs.umd.edu/trecis_2018/archived_text_sample_2013to2016_gensim_200.model.tgz)

