
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

def get_labels_dictionary(train_count, test_count):
    all_words = []
    train_labels = []
    test_labels = []
    with open('smss.txt') as smsfile:
        for i,line in enumerate(smsfile):
            if train_count != None:
                if (i == train_count + test_count):
                    break
            words = line.split()
            if words[0] == 'spam':
                if(i < train_count):
                    train_labels += [1]
                    all_words += words
                else:
                    test_labels += [1]
            if words[0] == 'ham':
                if (i < train_count):
                    train_labels += [0]
                    all_words += words
                else:
                    test_labels += [0]

    dictionary = Counter(all_words)
    del dictionary['spam']
    del dictionary['ham']

    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(len(dictionary))

    return train_labels,test_labels, dictionary

def extract_features(dictionary, start_index, end_index):
    features_matrix = np.zeros((end_index ,len(dictionary)))
    docID = 0
    with open('smss.txt') as datafile:
        for i,line in enumerate(datafile):
            if (i == end_index + start_index):
                break
            if (i >= start_index):
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

def main():
    # Create a dictionary of words with its frequency
    train_count = 1000
    test_count = 500
    train_labels, test_labels, dictionary = get_labels_dictionary(train_count, test_count)
    # Prepare feature vectors per training mail and its labels
    train_matrix = extract_features(dictionary,0, train_count)
    # Training Naive bayes classifier
    model = MultinomialNB()
    model.fit(train_matrix,train_labels)

    # Test the unseen mails for Spam
    test_matrix = extract_features(dictionary,train_count, test_count)
    result = model.predict(test_matrix)
    # for i in range(sms_count):
    #     if train_labels[i] != result[i]:
    #          with open('smss.txt') as datafile:
    #              for j,line in enumerate(datafile):
    #                  if (i == j):
    #                      print(line)
    #                      break
    print (confusion_matrix(test_labels,result))

main()
