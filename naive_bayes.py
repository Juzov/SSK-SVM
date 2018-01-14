
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import string_functions
import time



def get_labels_dictionary(train_count, test_count):
    all_words = []
    train_labels = []
    test_labels = []
    train_index = []
    test_index = []
    count_train_ham = 0
    count_test_ham = 0

    #Train
    with open('SMSSpamCollection') as smsfile:
        num_sms = sum(1 for line in open('SMSSpamCollection'))
        for i,line in enumerate(smsfile):
            words = line.split()
            if words[0] == 'ham':
                if (count_train_ham < train_count/2):
                    train_labels += [0]
                    all_words += words
                    count_train_ham += 1
                    train_index += [i]

            if words[0] == 'spam':
                if (i >= extract_spams(train_count/2)):
                    train_labels += [1]
                    all_words += words
                    train_index += [i]
    count_test_ham = 0
    count_test_spam = 0
    #Test
    with open('SMSSpamCollection') as smsfile:
        num_sms = sum(1 for line in open('SMSSpamCollection'))
        for i,line in enumerate(smsfile):
            words = line.split()
            if words[0] == 'ham':
                if i >= extract_hams(train_count / 2) and count_test_ham < test_count/2:
                    test_labels += [0]
                    all_words += words
                    count_test_ham += 1
                    test_index += [i]

            if words[0] == 'spam':
                if i >= extract_spams(train_count/2 + test_count/2) and count_test_spam < test_count/2:
                    test_labels += [1]
                    count_test_spam += 1
                    test_index += [i]
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

    return train_labels,test_labels, test_index, train_index, dictionary

def extract_features(dictionary, index_list):
    features_matrix = np.zeros((len(index_list), len(dictionary)))
    docID = 0
    with open('SMSSpamCollection') as datafile:
        for i,line in enumerate(datafile):
            if (i in index_list):
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

def extract_hams(amount_of_previous):
    ham_index = []
    with open('SMSSpamCollection') as datafile:
        for i,line in enumerate(datafile):
            if (line.split()[0] == 'ham'):
                ham_index.append(i)
    return ham_index[int(amount_of_previous)]

def extract_spams(amount_of_spams):
    spam_index = []
    with open('SMSSpamCollection') as datafile:
        for i,line in enumerate(datafile):
            if (line.split()[0] == 'spam'):
                spam_index.append(i)
    return spam_index[-int(amount_of_spams)]

def main():
    # Create a dictionary of words with its frequency
    train_count = 100
    test_count = 100
    #print(num_sms)
    global_start = time.time()
    print("Dictionary start")
    start = time.time()
    train_labels,test_labels, test_index, train_index, dictionary = get_labels_dictionary(train_count, test_count)
    end = time.time()
    print("Dictionary done", end - start)

    # Prepare feature vectors per training mail and its labels
    print("Extract Features training start")
    start = time.time()
    train_matrix = extract_features(dictionary,train_index)
    end = time.time()
    print("Extract features done", end - start)
    # Training Naive bayes classifier
    print("Naive Bayes start")
    start = time.time()
    model = MultinomialNB()
    model.fit(train_matrix,train_labels)
    end = time.time()
    print("Naive Bayes done", end - start)

    # Test the unseen mails for Spam
    print("Extract features test start")
    start = time.time()
    test_matrix = extract_features(dictionary,test_index)
    result = model.predict(test_matrix)
    end = time.time()
    print("Extract features test done", end - start)

    global_end = time.time()
    # for i in range(sms_count):
    #     if train_labels[i] != result[i]:
    #          with open('smss.txt') as datafile:
    #              for j,line in enumerate(datafile):
    #                  if (i == j):
    #                      print(line)
    #                      break
    print (confusion_matrix(test_labels,result))
    print("Execution time: ", global_end - global_start)

main()
