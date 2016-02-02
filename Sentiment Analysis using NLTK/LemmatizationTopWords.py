# NLP Homework NLTK
# Vibha Belavadi
# vxb141530

from nltk import FreqDist
from nltk.classify import util
from nltk.classify import MaxentClassifier
from nltk.corpus import movie_reviews
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import array

print 'Maximum Entropy classifier accuracy of raw data with lemmatization for top 1000 words with 25% of training data'

# Get the wordNet lemmatizer
lmtz = WordNetLemmatizer()

# Get the top 500 frequently occurring words in the movie_reviews
all_words = FreqDist(word for word in movie_reviews.words())
top_words = set(all_words.keys()[:1000])


# Dictionary of raw words:
def word_features(words):
    return dict([(lmtz.lemmatize(word), True) for word in top_words and words])


# Get all the movie reviews with positive data set and negative data set
negRev = movie_reviews.fileids('neg')
posRev = movie_reviews.fileids('pos')

# Mark the words in data set as positive and negative:
negWords = [(word_features(movie_reviews.words(fileids=[f])), 'neg') for f in negRev]
posWords = [(word_features(movie_reviews.words(fileids=[f])), 'pos') for f in posRev]

# Set cut off for separating the training data and the testing data:
posCutoff = len(posWords) * 25 / 100
negCutoff = len(negWords) * 25 / 100

# Fill the training data and the testing data with positive and negative data set:
TrainRev = posWords[:posCutoff] + negWords[:negCutoff]
Train_set = array(TrainRev)  # Using array of numpy for efficient calculation and more number of iterations
TestRev = posWords[posCutoff:] + negWords[negCutoff:]
Test_set = array(TestRev)  # Using array of numpy for efficient calculation and more number of iterations

print 'Training on %d instances and testing on %d instances' % (len(Train_set), len(Test_set))

# Call Maximum Entropy classifier to classify the training data:
algo = MaxentClassifier.ALGORITHMS[0]
classifier = MaxentClassifier.train(Train_set, algorithm=algo, max_iter=3)
classifier.show_most_informative_features(10)

# Print the algorithm accuracy
print 'The accuracy is', util.accuracy(classifier, Test_set)
