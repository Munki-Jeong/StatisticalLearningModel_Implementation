import os
from decimal import Decimal
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import glob

#Read the data
ham_dir = "./data/myham/"
spam_dir = "./data/myspam/"

#text preprocessing
def preprocess(directory):
    files = os.listdir(directory)
    v = CountVectorizer(input='filename')
    matrix = v.fit_transform([os.path.join(directory, file) for file in files])
    word_occurrences = matrix.sum(axis=0).A1
    word_occurrences_dict = {word: word_occurrences[idx] for word, idx in v.vocabulary_.items()}
    return word_occurrences_dict

ham_word_occurrences_dict = preprocess(ham_dir)
spam_word_occurrences_dict = preprocess(spam_dir)


class NB:
    def __init__(self, dict1):
        self.dict1 = dict1
        self.total_count = sum(list(dict1.values()))
    def get_ll(self, word, dict1_laplace_smoothing_dict = None):
        if dict1_laplace_smoothing_dict == None:
            return Decimal(self.dict1[word]/self.total_count)
        else:

            return Decimal(dict1_laplace_smoothing_dict[word]/sum(list(dict1_laplace_smoothing_dict.values())))

    def laplace_smoothing(self, test_dict, dict1_laplace_smoothing_dict, alpha = 1 ):
        for word in test_dict.keys(): #test data에 존재하는 단어 중, ham, spam 문서에는 존재하지 않았던 단어들을 ham, spam_ll_dictionary에 추가해줌
            if word not in dict1_laplace_smoothing_dict.keys():
                dict1_laplace_smoothing_dict[word] = 0
        for word in dict1_laplace_smoothing_dict.keys():    #그리고 그 단어들을 포함해 모든 단어들이 각각 alpha(here, alpha=1)번씩 더 있다고 생각하고 더해줌
            dict1_laplace_smoothing_dict[word] += alpha
        return dict1_laplace_smoothing_dict




#2. train
# to-do: 1)prior 2)likelihood of each feature(here, word)

ham_inst = NB(ham_word_occurrences_dict)
spam_inst = NB(spam_word_occurrences_dict)

#2-1)prior
ham_files = os.listdir(ham_dir)
spam_files = os.listdir(spam_dir)
ham_prior = len(ham_files) / (len(ham_files)+len(spam_files))
spam_prior = len(spam_files) / (len(ham_files)+len(spam_files))

#2-2)likelihood for each words
ham_ll = ham_word_occurrences_dict
spam_ll = spam_word_occurrences_dict


ham_ll = {word: ham_inst.get_ll(word) for word in ham_ll.keys()}
spam_ll = {word: spam_inst.get_ll(word) for word in spam_ll.keys()}


#3)Predict
test_dir ="./data/mytest"
test_word_occurrences_dict = preprocess(test_dir)

ham_post = Decimal(ham_prior)
spam_post = Decimal(spam_prior)

alpha = 1

# Calculate posterior probabilities for ham and spam with laplace smoothing
ham_laplace_ll_dict = {key: value for key, value in ham_word_occurrences_dict.items()}
spam_laplace_ll_dict = {key: value for key, value in spam_word_occurrences_dict.items()}

ham_laplace_ll_dict = ham_inst.laplace_smoothing(test_word_occurrences_dict, ham_laplace_ll_dict, alpha)
spam_laplace_ll_dict = spam_inst.laplace_smoothing(test_word_occurrences_dict, spam_laplace_ll_dict, alpha)


ham_laplace_ll_dict = {word: ham_inst.get_ll(word, ham_laplace_ll_dict) for word in list(ham_laplace_ll_dict.keys())}
spam_laplace_ll_dict = {word: spam_inst.get_ll(word, spam_laplace_ll_dict) for word in list(spam_laplace_ll_dict.keys())}

for word, count in test_word_occurrences_dict.items():
    ham_post *= ham_laplace_ll_dict[word] ** count
    spam_post *= spam_laplace_ll_dict[word] ** count

# spam_post to predict the class
if ham_post > spam_post:
    prediction = "ham"
else:
    prediction = "spam"
print(f"Prediction: {prediction}")
print(ham_post, spam_post)

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

ham_files = glob.glob(os.path.join(ham_dir, "*.txt"))
ham_text = [read_file(file) for file in ham_files]
ham_label = [0] * len(ham_text)

spam_files = glob.glob(os.path.join(spam_dir, "*.txt"))
spam_text = [read_file(file) for file in spam_files]
spam_label = [1] * len(spam_text)

train_data = ham_text + spam_text
y_train = ham_label + spam_label

test_file = os.listdir(test_dir)
test_data = [read_file(os.path.join(test_dir, file)) for file in test_file]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
print(nb_predictions)

posterior_probabilities = nb_classifier.predict_proba(X_test)
ham_posterior_probabilities = posterior_probabilities[:, 0]
spam_posterior_probabilities = posterior_probabilities[:, 1]
print(ham_posterior_probabilities, spam_posterior_probabilities)
