import numpy as np

import nltk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from src import Constants


class PosTokenizer(object):

    def __call__(self, doc):
        tokenizer = nltk.word_tokenize(doc)
        tags = [item for (_, item) in nltk.pos_tag(tokenizer)]
        #print("doc: %s, tags: %s" % (doc, tags))
        return tags


class MixTokenizer(object):
    def __call__(self, doc):
       pass

# dict_classifiers = {
#     "Logistic Regression": LogisticRegression(penalty='l1',
#                                         dual=False,
#                                         tol=0.000001,
#                                         C=0.5,
#                                         fit_intercept=True,
#                                         intercept_scaling=1,
#                                         class_weight=None,
#                                         random_state=None,
#                                         solver='saga',
#                                         max_iter=10000,
#                                         multi_class='multinomial',
#                                         verbose=0,
#                                         warm_start=False,
#                                         n_jobs=1),
#     "Nearest Neighbors": KNeighborsClassifier(),
#     "Linear SVM": LinearSVC(),
#       "Baese": MultinomialNB(),
#     "Gradient Boosting Classifier": GradientBoostingClassifier(),
#     "Decision Tree": nltk.DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators = 18),
#     "Neural Net": MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.00001),
#     "Naive Bayes": GaussianNB()
# }


def get_metrics(true_labels, predicted_labels):
    # print true_labels, predicted_labels
    ac = np.round(accuracy_score(true_labels, predicted_labels), 2)
    pre = np.round(precision_score(true_labels, predicted_labels, average='weighted'), 2)
    rec = np.round(recall_score(true_labels, predicted_labels, average='weighted'), 2)
    f1 = np.round(f1_score(true_labels, predicted_labels,average='weighted'), 2)
    f1_macro = np.round(f1_score(true_labels, predicted_labels, average='macro'), 2)
    return [ac, pre, rec, f1, f1_macro]

def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    scores = get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions, scores

def print_dictionary(d):
    for key, value in d.items():
        print "name: %s \n" % key
        print str(value)

vector_map = {
    'tf-idf': TfidfVectorizer(analyzer='word', tokenizer=PosTokenizer(), ngram_range=(1, 3), norm=None, smooth_idf=False, sublinear_tf=True, lowercase=False, max_features=1000),
    'tf': TfidfVectorizer(analyzer='word', tokenizer=PosTokenizer(), ngram_range=(1, 3), norm=None, smooth_idf=False, sublinear_tf=True, lowercase=False, max_features=1000, use_idf=False),
    'hash': HashingVectorizer(analyzer='word', tokenizer=PosTokenizer(), ngram_range=(1, 3)),
    'count': CountVectorizer(analyzer='word', tokenizer=PosTokenizer(), ngram_range=(1, 3))
}

def classify(data, classifers, vectorizer):
    count_examples = len(data.target)
    count_classes = len(data.target_names)
    print("Examples %s. Classes %s" %(count_examples, count_classes))
    confusions = {}
    correct_answers = {}
    accurasy = {}
    for name in classifers.keys():
        confusions[name] = np.zeros((count_classes, count_classes))
        correct_answers[name] = 0
        accurasy[name] = []
    k_fold = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in k_fold.split(data.data):
        train_data = [data.data[d] for d in train_index]
        train_vectors = vectorizer.fit_transform(train_data)
        train_labels = [data.target[d] for d in train_index]
        test_data = [data.data[d] for d in test_index]
        test_vectors = vectorizer.transform(test_data)
        test_labels = [data.target[d] for d in test_index]
        score = {}
        for name, cl in classifers.items():
            p, scores = train_predict_evaluate_model(cl, train_vectors, train_labels, test_vectors, test_labels)
            accurasy[name].append(scores)
            correct_answers[name] += accuracy_score(test_labels, p, normalize=False)
            confusions[name] += confusion_matrix(test_labels, p, labels=list((set(train_labels))))
    print_dictionary(confusions)
    print correct_answers
    for name, acc in accurasy.items():
        print "name: %s, %s" % (name, np.average(acc, axis=0))



path = Constants.test
data = load_files(path)
cl = {}
cl["Logistic Regression"] = LogisticRegression()
cl["LinearSVM"] = LinearSVC(loss='l1')
cl["Bayes"] = MultinomialNB()
cl["Neural Net"] = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', alpha=0.0001)
# cl["k-nearest"] = KNeighborsClassifier()
# cl["Desision tree"] = nltk.DecisionTreeClassifier()
# cl["Gradient Boosting Classifier"] =  GradientBoostingClassifier()
# cl["Random Forest"] = RandomForestClassifier()
vectorizer = TfidfVectorizer(analyzer='word', tokenizer=PosTokenizer(), ngram_range=(1, 3), norm=None, smooth_idf=False, sublinear_tf=True, lowercase=False, max_features=1000, use_idf=False)
classify(data, cl, vectorizer)
