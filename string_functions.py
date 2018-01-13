import os
import nltk as nltk
from nltk.corpus import stopwords
from nltk.corpus import reuters
import re
import numpy as np
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
    path = os.path.dirname(os.path.abspath(__file__))

    with open(path + '/SMSSpamCollection', 'r') as email:
        content = email.readlines()

    spam_data = []
    ham_data = []

    for i, message in enumerate(content):
        message = message.strip()
        if("spam" in message):
            message = message.replace('spam','')
            spam_data.append(format_text(message))
        else:
            message = message.replace('ham','')
            ham_data.append(format_text(message))

    test_spam = spam_data[:int(len(spam_data)*0.25)]
    train_spam = spam_data[int(len(spam_data)*0.25):]
    test_ham = ham_data[:int(len(ham_data)*0.25)]
    train_ham = ham_data[int(len(ham_data)*0.25):]

    test_data = test_ham + test_spam
    train_data = train_ham + train_spam

    train_labels_ham = np.zeros(int(len(train_ham))).tolist()
    train_labels_spam = np.ones(int(len(train_spam))).tolist()
    train_labels = train_labels_ham + train_labels_spam

    test_labels_ham = np.zeros(int(len(test_ham))).tolist()
    test_labels_spam = np.ones(int(len(test_spam))).tolist()
    test_labels = test_labels_ham + test_labels_spam

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
