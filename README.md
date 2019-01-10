# Overview

This repository contains a set of experiments for the [Incident Streams](http://www.trecis.org) track at [TREC 2018](https://trec.nist.gov/).

We leverage machine learning models of increasing complexity for classifying Twitter content according to an ontology of 23 crisis responder information needs. 
Results demonstrate a simple set of one-versus-rest naive Bayes classifiers significantly outperforms a random baseline by an order of magnitude, but performance is still low (accuracy <50%).
Enriching the dataset with FastText-based word vectors built from Twitter achieves only marginal performance increases, and expanding the training data through label propagation reduced performance.

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

We train our models using two datasets, the first of which is provided by the TREC-IS organizers and contains 1,261 rehydrated tweets across 23 classes, as shown in Table \ref{tab:class_dist}.
Additionally, we create a sample, to which we will refer as the PROP set, of 741,859 tweets from Twitter's public sample stream between 2013 and 2016 that contain disaster-related vocabulary words identified by Buntain and Lim \cite{Buntain2018}.
For the TREC-IS dataset, labels are provided by the track organizers; for the PROP set, however, we rely on scikit-learn's label spreading library.
Label spreading uses a distance metric to propagate class labels from known instances to similar but unlabeled instances.
In this implementation, we use the $k$ nearest neighbors metric, requiring at least $5$ manually labeled neighbors in our metric space to be "close" to an unlabeled sample to propagate the manual label.
For scalability reasons, we downsample this PROP set to 20% of the total and, in Scenario 2, we use singular value decomposition to reduce dimensionality to 10 components.

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
You can download this vectorizer here: [2013to2016_tfidf_vectorizer_20190109.pkl](http://obj.umiacs.umd.edu/trecis_2018/2013to2016_tfidf_vectorizer_20190109.pkl)

We also provide our FastText-trained model on this same set of English tweets, which you can find here: [archived_text_sample_2013to2016_gensim_200.model.tgz](http://obj.umiacs.umd.edu/trecis_2018/archived_text_sample_2013to2016_gensim_200.model.tgz)

