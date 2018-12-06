# coding:utf-8
from sklearn.linear_model import LogisticRegression as LR
from config import id2label
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from gensim import models


def read_file(read_path):
    """read files"""
    temp = open(read_path, 'r')
    content = temp.readlines()
    temp.close()
    return content


def write_cutted_corpus(content, write_path):
    """write files"""
    temp = open(write_path, 'w')
    temp.write("\n".join(content))
    temp.close()


def cut_files(content):
    """"use jieba to cut the files"""
    token_list = []
    for row in content:
        items = row.strip().split("	", 1)
        tokens = list(jieba.cut(items[1]))
        token_row = " ".join(tokens)
        token_list.append(token_row)
    return token_list


def cut_files_labels(content):
    """"use jieba to cut the files"""
    labels_list = []
    for row in content:
        items = row.strip().split("	", 1)
        labels = list(jieba.cut(items[0]))
        labels_row = " ".join(labels)
        labels_list.append(labels_row)
    return labels_list


def preprocessing(read_path, write_path, write_path_labels):
    """"use read_file(path) and cut_files(content) and
    write_cutted_corpus(content,write_path) to do the
    prepocessing for tfidf"""
    content = read_file(read_path)
    content = cut_files(content)
    write_cutted_corpus(content, write_path)
    labels = read_file(read_path)
    labels = cut_files_labels(labels)
    write_cutted_corpus(labels, write_path_labels)


def tfidf(train_content, test_content, max_features, stop_words):
    """use tfidf to get data"""
    tfidf_model = TfidfVectorizer(stop_words=stop_words, max_features=max_features).fit(train_content)
    X_train = tfidf_model.transform(train_content)
    X_test = tfidf_model.transform(test_content)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    return X_train, X_test


def label2id(label_path):
    labels = read_file(label_path)
    labels_ = []
    for num in id2label:
        for row in labels:
            item = row.strip()
            if item == id2label[num]:
                labels_.append(num)
    return labels_


def train_LR(X_train, Y_train):
    """LogisticRegression train"""
    logistic = LR()
    logistic.fit(X_train, Y_train)
    return logistic


def test_LR(model, X_test):
    """LogisticRegression test"""
    result = model.predict(X_test)
    print("the predict of LogisticRegression are:"),
    print (list(result))


def score_LR(model, X_test, Y_test):
    """get the score of LogisticRegression"""
    score = model.score(X_test, Y_test)
    print("the accurate of the LogisticRegression is:"),
    print(score)


def train_NB(X_train, Y_train):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    return model


def test_NB(model, X_test):
    result = model.predict(X_test)
    print("the predict of NaiveBayes are:"),
    print(list(result))


def score_NB(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    print("the accurate of the NaiveBayes is:"),
    print(score)


def train_SVMC(X_train, Y_train):
    model = LinearSVC()
    model.fit(X_train, Y_train)
    return model


def test_SVMC(model, X_test):
    result = model.predict(X_test)
    print("the predict of SVMC are:"),
    print(list(result))


def score_SVMC(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    print("the accurate of the SVMC is:"),
    print(score)


def train_SVMR(X_train, Y_train):
    model = LinearSVR()
    model.fit(X_train, Y_train)
    return model


def test_SVMR(model, X_test):
    result = model.predict(X_test)
    print("the predict of SVMR are:"),
    print(list(result))


def score_SVMR(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    print("the accurate of the SVMR is:"),
    print(score)

    
def train_DTCl(X_train, Y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    return model


def test_DTCl(model, X_test):
    result = model.predict(X_test)
    print("the predict of DecisionTreeClassifier are:"),
    print(list(result))


def score_DTCl(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    print("the accurate of the DecisionTreeClassifier is:"),
    print(score)


def train_DTRe(X_train, Y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, Y_train)
    return model


def test_DTRe(model, X_test):
    result = model.predict(X_test)
    print("the predict of DecisionTreeRegressor are:"),
    print(list(result))


def score_DTRe(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    print("the accurate of the DecisionTreeRegressor is:"),
    print(score)


def train_NNCl(X_train, Y_train):
    model = MLPClassifier()
    model.fit(X_train, Y_train)
    return model


def test_NNCl(model, X_test):
    result = model.predict(X_test)
    print("the predict of MLPClassifier are:"),
    print(list(result))


def score_NNCl(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    print("the accurate of the MLPClassifier is:"),
    print(score)


def train_NNRe(X_train, Y_train):
    model = MLPRegressor()
    model.fit(X_train, Y_train)
    return model


def test_NNRe(model, X_test):
    result = model.predict(X_test)
    print("the predict of MLPRegressor are:"),
    print(list(result))


def score_NNRe(model, X_test, Y_test):
    score = model.score(X_test, Y_test)
    print("the accurate of the MLPRegressor is:"),
    print(score)


if __name__=="__main__":
    """the preprocessing of text"""
    #preprocessing("./cnews.test.txt", "./cnews.test.tfidf.txt", "./cnews.test.tfidf.labels.txt")
    #preprocessing("./cnews.train.txt", "./cnews.train.tfidf.txt", "./cnews.train.tfidf.labels.txt")

    X_train, X_test = tfidf(read_file("./cnews.train.tfidf.txt"), read_file("./cnews.test.tfidf.txt"), 500,
                            read_file("./hlt_stop_words.txt"))
    Y_test = label2id("./cnews.test.tfidf.labels.txt")
    Y_train = label2id("./cnews.train.tfidf.labels.txt")

    """LogisticRegression"""
    logistic_model = train_LR(X_train, Y_train)
    test_LR(logistic_model, X_test)
    score_LR(logistic_model, X_test, Y_test)
    print(" ")


    """NaiveBayes"""
    NB_model = train_NB(X_train, Y_train)
    test_NB(NB_model, X_test)
    score_NB(NB_model, X_test, Y_test)
    print(" ")

    """SVM"""
    SVMC_model = train_SVMC(X_train, Y_train)
    test_SVMC(SVMC_model, X_test)
    score_SVMC(SVMC_model, X_test, Y_test)
    print(" ")
    SVMR_model = train_SVMR(X_train, Y_train)
    test_SVMR(SVMR_model, X_test)
    score_SVMR(SVMR_model, X_test, Y_test)
    print(" ")

    """DecisionTree"""
    DTCl_model = train_DTCl(X_train, Y_train)
    test_DTCl(DTCl_model, X_test)
    score_DTCl(DTCl_model, X_test, Y_test)
    print(" ")
    DTRe_model = train_DTRe(X_train, Y_train)
    test_DTRe(DTRe_model, X_test)
    score_DTRe(DTRe_model, X_test, Y_test)
    print(" ")

    """NeuralNetwork"""
    NNCl_model = train_NNCl(X_train, Y_train)
    test_NNCl(NNCl_model, X_test)
    score_NNCl(NNCl_model, X_test, Y_test)
    print(" ")
    NNRe_model = train_NNRe(X_train, Y_train)
    test_NNRe(NNRe_model, X_test)
    score_NNRe(NNRe_model, X_test, Y_test)
    print(" ")