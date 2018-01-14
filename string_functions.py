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


def get_info(is_spam, amount_of_test_documents, amount_of_train_documents):
    test_docs = None
    train_docs = None
    train_labels = None
    test_labels = None
    if(is_spam):
        test_docs, train_docs, train_labels, test_labels = get_spam()
    else:
        test_docs, train_docs, train_labels, test_labels = get_reuters()
    
    first_and_last_test = int(amount_of_test_documents * 0.5)
    first_and_last_train = int(amount_of_train_documents * 0.5)

    test_docs = test_docs[: first_and_last_test] + \
        test_docs[- first_and_last_test:]
    train_docs = train_docs[: first_and_last_train] + \
        train_docs[- first_and_last_train:]
    test_labels = test_labels[: 
        first_and_last_test] + test_labels[- first_and_last_test:]
    train_labels = train_labels[: 
        first_and_last_train] + train_labels[- first_and_last_train:]

    return test_docs, train_docs, train_labels, test_labels


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


# Get spam corpus


def get_spam():
    path = os.path.dirname(os.path.abspath(__file__))

    with open(path + '/SMSSpamCollection', 'r') as email:
        content = email.readlines()

    spam_data = []
    ham_data = []

    for i, message in enumerate(content):
        message = message.strip()
        if("spam" in message):
            message = message.replace('spam', '')
            spam_data.append(format_text(message))
        else:
            message = message.replace('ham', '')
            ham_data.append(format_text(message))

    test_spam = spam_data[:int(len(spam_data) * 0.5)]
    train_spam = spam_data[int(len(spam_data) * 0.5):]
    test_ham = ham_data[:int(len(ham_data) * 0.5)]
    train_ham = ham_data[int(len(ham_data) * 0.5):]

    test_data = test_ham + test_spam
    train_data = train_ham + train_spam

    train_labels_ham = np.zeros(int(len(train_ham))).tolist()
    train_labels_spam = np.ones(int(len(train_spam))).tolist()
    train_labels = train_labels_ham + train_labels_spam

    test_labels_ham = np.zeros(int(len(test_ham))).tolist()
    test_labels_spam = np.ones(int(len(test_spam))).tolist()
    test_labels = test_labels_ham + test_labels_spam

    return test_data, train_data, train_labels, test_labels


def get_most_used(word_amount, k, train_docs):
    d = {}
    string_length = k

    for i, text in enumerate(train_docs):
        for j in range(0, len(text) - string_length):
            if text[j:j + string_length] in d:
                d[text[j:j + string_length]] += 1
            else:
                d[text[j:j + string_length]] = 1

    most_used = sorted(d.items(), key=lambda x: x[1])
    most_used = most_used[len(most_used) - word_amount:len(most_used)]

    return most_used
