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
    path = os.path.dirname(os.path.abspath(__file__))

    with open(path + '/SMSSpamCollection', 'r') as email:
        content = email.readlines()

    test_data = []
    train_data = []
    test_labels = []
    train_labels = []

    for i, message in enumerate(content):
        message = message.strip()
        if(i < len(content)/2):
            if("spam" in message):
                train_labels.append(1)
                message = message.replace('spam','')
                train_data.append(format_text(message))
            else:
                train_labels.append(0)
                message = message.replace('ham','')
                train_data.append(format_text(message))

        else:
            if("spam" in message):
                test_labels.append(1)
                message = message.replace('spam','')
                test_data.append(format_text(message))
            else:
                test_labels.append(0)
                message = message.replace('ham','')
                test_data.append(format_text(message))

    test_data = list(filter(None, test_data))
    train_data = list(filter(None, train_data))
    train_labels = list(filter(None, train_labels))
    test_labels = list(filter(None, test_labels))

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
