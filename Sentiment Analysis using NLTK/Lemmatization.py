from nltk.classify import util
from nltk.classify import MaxentClassifier
from nltk.corpus import movie_reviews
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import array

print "Maximum Entropy classifier accuracy of raw data with Lemmatization"

# Get the wordNet lemmatizer
lmtz = WordNetLemmatizer()


# Dictionary of raw words:
def word_features(words):
    return dict([(lmtz.lemmatize(word), True) for word in words])


# Get all the movie reviews with positive data set and negative data set
posRev = movie_reviews.fileids('pos')
negRev = movie_reviews.fileids('neg')

# Mark the words in data set as positive and negative:
posWords = [(word_features(movie_reviews.words(fileids=[f])), 'pos') for f in posRev]
negWords = [(word_features(movie_reviews.words(fileids=[f])), 'neg') for f in negRev]

# Set cut off for separating the training data and the testing data:
posCutoff = len(posWords) * 25 / 100
negCutoff = len(negWords) * 25 / 100

# Fill the training data and the testing data with positive and negative data set:
TestRev = posWords[posCutoff:] + negWords[negCutoff:]
Test_set = array(TestRev)
TrainRev = posWords[:posCutoff] + negWords[:negCutoff]
Train_set = array(TrainRev)
print 'train on %d instances, test on %d instances' % (len(Train_set), len(Test_set))

# Call Maximum Entropy classifier to classify the training data:
algo = MaxentClassifier.ALGORITHMS[0]
classifier = MaxentClassifier.train(Train_set, algorithm=algo, max_iter=3)
classifier.show_most_informative_features(10)

# Print the algorithm accuracy
print 'Accuracy is', util.accuracy(classifier, Test_set)
