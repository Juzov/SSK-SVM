
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def make_Dictionary(train_dir):
    emails = []
    for f in os.listdir(train_dir):
        emails.append(os.path.join(train_dir,f))
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i != 1:
                    words = line.split()
                    print("Words ", words)
                    all_words += words

    dictionary = Counter(all_words)

    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    
    dictionary = dictionary.most_common(1000)
    # dictionary = list(dictionary.items())
    
    # global key
    # new_dictionary = []
    # for item,curr_key in dictionary:
    #     if len(item) == key:
    #         new_dictionary.append(item)
    return dictionary

def extract_features(train_dir, dictionary):
    emails = []
    for f in os.listdir(train_dir):
        emails.append(os.path.join(train_dir,f))
    features_matrix = np.zeros((len(emails),len(dictionary)))
    docID = 0;
    for mail in emails:
      with open(mail) as m:
        for i,line in enumerate(m):
          if i != 1:
            words = line.split()
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
    train_dir = 'ling-spam/train-mails'
    dictionary = make_Dictionary(train_dir)
    #print(len(dictionary))

    exit()
    
    # Prepare feature vectors per training mail and its labels

    train_labels = np.zeros(702)
    train_labels[351:701] = 1
    train_matrix = extract_features(train_dir, dictionary)

    # Training SVM and Naive bayes classifier and its variants

    model = MultinomialNB()
    model.fit(train_matrix,train_labels)

    # Test the unseen mails for Spam

    test_dir = 'ling-spam/test-mails'
    test_matrix = extract_features(test_dir, dictionary)
    test_labels = np.zeros(260)
    test_labels[130:260] = 1

    result = model.predict(test_matrix)

    print (confusion_matrix(test_labels,result))

main()
