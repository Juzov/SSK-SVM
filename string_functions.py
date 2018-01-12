import os
import nltk as nltk
from nltk.corpus import stopwords
from nltk.corpus import reuters
import re

# Remove stopwords and symbols


def format_text(text):
    pattern = re.compile(
        r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text.lower()
    textWihoutStopWords = pattern.sub('', text)
    textWihoutSymbols = re.sub(r'[^a-zA-Z\d\s]', '', textWihoutStopWords)
    formattedText = re.sub(r"\s+", " ", textWihoutSymbols)
    return formattedText

# Get the reuters corpus

def get_info(is_spam):
	if(is_spam):
		return get_spam()
	else:
		return get_reuters()

def get_reuters():
    documents = reuters.fileids()

    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                               documents))

    # train_docs_id = train_docs_id[:2]
    # test_docs_id = test_docs_id[:2]

    # train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    # test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

    train_docs = [format_text(reuters.raw(doc_id)) for doc_id in train_docs_id]
    test_docs = [format_text(reuters.raw(doc_id)) for doc_id in test_docs_id]

    train_labels = [reuters.categories(doc_id)
                    for doc_id in train_docs_id]
    test_labels = [reuters.categories(doc_id)
                   for doc_id in test_docs_id]

    return test_docs, train_docs, train_labels, test_labels


#Get spam corpus


def get_spam():
    path = os.path.dirname(os.path.abspath(__file__)) + '/ling-spam/'
    path_test = path + 'test-mails/'
    path_train = path + 'train-mails/'
    test_data = [None] * len(os.listdir(path_test))
    train_data = [None] * len(os.listdir(path_train))
    test_labels = [None] * len(os.listdir(path_test))
    train_labels = [None] * len(os.listdir(path_train))

    for i, filename in enumerate(os.listdir(path_test)):
        with open(path_test + filename, 'r') as email:
            test_data[i] = format_text(email.read())
        if("spmsg" in filename):
            test_labels[i] = 1
        else:
            test_labels[i] = 0

    for i, filename in enumerate(os.listdir(path_train)):
        with open(path_train + filename, 'r') as email:
            train_data[i] = format_text(email.read())
        if("spmsg" in filename):
            train_labels[i] = 1
        else:
            train_labels[i] = 0

    return test_data, train_data, train_labels, test_labels

def get_most_used(is_spam):
    d = {}
    string_length = 5
    test_data, train_data, train_labels, test_labels = get_info(is_spam)

    for i, text in enumerate(train_data):
        for j in range(0, len(text) - string_length):
            if text[j:j + string_length] in d:
                d[text[j:j + string_length]] += 1
            else:
                d[text[j:j + string_length]] = 1

    most_used = sorted(d.items(), key=lambda x: x[1])
    most_used = most_used[len(most_used) - 200:len(most_used)]

    return most_used
