
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

# def make_Dictionary(train_dir):
#     emails = []
#     for f in os.listdir(train_dir):
#         emails.append(os.path.join(train_dir,f))
#     all_words = []
#     for mail in emails:
#         with open(mail) as m:
#             for i,line in enumerate(m):
#                 if i != 1:
#                     words = line.split()
#                     all_words += words
#
#     dictionary = Counter(all_words)
#
#     list_to_remove = dictionary.keys()
#     for item in list(list_to_remove):
#         if item.isalpha() == False:
#             del dictionary[item]
#         elif len(item) == 1:
#             del dictionary[item]
#     dictionary = dictionary.most_common(100)
#     return dictionary

#
#
# def extract_features(train_dir, dictionary):
#     emails = []
#     for f in os.listdir(train_dir):
#         emails.append(os.path.join(train_dir,f))
#     features_matrix = np.zeros((len(emails),100))
#     docID = 0;
#     for mail in emails:
#       with open(mail) as m:
#         for i,line in enumerate(m):
#           if i != 1:
#             words = line.split()
#             for word in words:
#               wordID = 0
#               for i,d in enumerate(dictionary):
#                 if d[0] == word:
#                   wordID = i
#                   features_matrix[docID,wordID] = words.count(word)
#         docID = docID + 1
#     return features_matrix

def get_labels_dictionary():
    all_words = []
    sms_labels = []
    with open('smss.txt') as smsfile:
        for i,line in enumerate(smsfile):
            words = line.split()
            if words[0] == 'spam':
                sms_labels += [1]
            if words[0] == 'ham':
                sms_labels += [0]
            all_words += words

    dictionary = Counter(all_words)
    del dictionary['spam']
    del dictionary['ham']

    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(2000)

    return sms_labels, dictionary

def extract_features(dictionary, sms_count):
    features_matrix = np.zeros((sms_count,2000))
    docID = 0
    with open('smss.txt') as datafile:
        for i,line in enumerate(datafile):
            words = line.split()
            del words[0]
            for word in words:
                wordID = 0
                for i,d in enumerate(dictionary):
                    if d[0] == word:
                        wordID = i
                        features_matrix[docID,wordID] = words.count(word)
            docID = docID + 1
    return features_matrix

#Global variable for the length of substrings
#key = 4
def main():
    # Create a dictionary of words with its frequency
    train_labels, dictionary = get_labels_dictionary()
    sms_count = len(train_labels)
    # Prepare feature vectors per training mail and its labels
    train_matrix = extract_features(dictionary, sms_count)
    # Training Naive bayes classifier
    model = MultinomialNB()
    model.fit(train_matrix,train_labels)

    # Test the unseen mails for Spam
    test_matrix = extract_features(dictionary, sms_count)
    result = model.predict(test_matrix)

    print (confusion_matrix(train_labels,result))

main()
