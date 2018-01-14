import sys
import os
from sklearn import svm
import sklearn.preprocessing as preprocessing
import nltk as nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
import numpy as np
import math
import re
import ssk

"Function for removing stopwords and symbols"
def format_text(text):
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	text.lower()
	textWihoutStopWords = pattern.sub('', text)
	textWihoutSymbols = re.sub(r'[^a-zA-Z\d\s]','', textWihoutStopWords)
	formattedText = re.sub(r"\s+", " ", textWihoutSymbols)
	return formattedText

"Get the reuters corpus"
def get_reuters():
	documents = reuters.fileids()
 
	train_docs_id = list(filter(lambda doc: doc.startswith("train"),
	                            documents))
	test_docs_id = list(filter(lambda doc: doc.startswith("test"),
	                       documents))

	train_docs_id = train_docs_id[:2]
	test_docs_id = test_docs_id[:2]

	train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
	test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

	train_docs = [format_text(reuters.raw(doc_id)) for doc_id in train_docs_id]
	test_docs = [format_text(reuters.raw(doc_id)) for doc_id in test_docs_id]

	train_labels = [reuters.categories(doc_id)
	                                  for doc_id in train_docs_id]
	test_labels = [reuters.categories(doc_id)
	                             for doc_id in test_docs_id]

	return test_docs, train_docs, train_labels, test_labels

"Get spam corpus"
def get_spam():
	path = os.path.dirname(os.path.abspath(__file__)) + '/ling-spam/'
	path_test = path + 'test-mails/'
	path_train = path + 'train-mails/'
	test_data = [None]*len(os.listdir(path_test))
	train_data = [None]*len(os.listdir(path_train))
	test_labels = [None]*len(os.listdir(path_test))
	train_labels = [None]*len(os.listdir(path_train))

	for i, filename in enumerate(os.listdir(path_test)):
		with open(path_test+filename,'r') as email:
			test_data[i] = format_text(email.read()[:450])
		if("spmsgc" in filename):
			test_labels[i] = 1
		else:
			test_labels[i] = 0

	for i, filename in enumerate(os.listdir(path_train)):
		with open(path_train+filename,'r') as email:
			train_data[i] = format_text(email.read()[:450])
		if("spmsgc" in filename):
			train_labels[i] = 1
		else:
			train_labels[i] = 0

	return test_data, train_data, train_labels, test_labels

sys.setrecursionlimit(100000)
k = 1

#test_docs, train_docs, train_labels, test_labels = get_reuters()
test_docs, train_docs, train_labels, test_labels = get_spam()

print("Length test_doc ", len(test_docs))
print("Length train_doc ",len(train_docs))
print("Length test_label ",len(test_labels))
print("Length train_label ",len(train_labels))

l = 100

#Fixing the wrong positions of files caused by Linux
index_test_labels = np.argsort(test_labels)
index_train_labels = np.argsort(train_labels)

print("Index train labels ", train_labels)
print("Index test labels ", index_train_labels)
print("Test_labels ", test_labels)
print("Index test label ",index_test_labels)

new_test_labels = []
new_test_docs = []
new_train_labels = []
new_train_docs = []
for i in range(len(train_labels)):
	if (i < len(test_labels)):
		new_test_labels.append(test_labels[index_test_labels[i]])
		new_test_docs.append(test_docs[index_test_labels[i]])
	new_train_labels.append(train_labels[index_train_labels[i]])
	new_train_docs.append(train_docs[index_train_labels[i]])

train_docs = new_train_docs
test_docs = new_test_docs
train_labels = new_train_labels
test_labels = new_test_labels

print("Train labels ", train_labels)
print("Test labels ", test_labels)

# print("Index 0 ", test_labels[index_test_labels[0]])
# print("Index 1 ", test_labels[index_test_labels[1]])

# divider = int(l/2)

# index_test_labels[divider:] = -1
# index_train_labels[divider:] = -1

# for i in range(l):	 
# 	j = 0
# 	while(j < len(train_labels)):
# 		if i  < l /2:
# 			if (j < len(test_labels)):
# 				if test_labels[j] == 0:
# 					if (j in index_test_labels) == False:
# 						index_test_labels[i] = j
# 			if train_labels[j] == 0:
# 				if (j in index_train_labels) == False:
# 					index_train_labels[i] = j	
# 		else:
# 			if (j < len(test_labels)):
# 				if test_labels[j] == 1:
# 					if (j in index_test_labels) == False:
# 						index_test_labels[i] = j
# 			if train_labels[j] == 1:
# 				if (j in index_train_labels) == False:
# 					index_train_labels[i] = j				

# 		j += 1	

	#print(test_docs[index_test_docs[i]])
	#print(train_docs[index_train_docs[i]])
#print("index : ", index_test_labels)
#print("index : ", index_train_labels)
	#print(test_labels[index_test_labels[i]])
	#print(train_labels[index_train_labels[i]])

# test_docs = test_docs[:5]+test_docs[255:]
# train_docs = train_docs[:5]+train_docs[697:]
# test_labels = test_labels[:5]+test_labels[255:]
# train_labels = train_labels[:5]+train_labels[697:]

# print("Length test_doc ",test_docs)
# print("Length train_doc ",train_docs)
# print("Length test_label ",test_labels)
# print("Length train_label ",train_labels)



gram = np.zeros((len(train_docs),len(train_docs)))
for i in range(0,len(train_docs)):
	for j in range(i, len(train_docs)):
		gram[i][j] = ssk.getSSK(train_docs[i],train_docs[j], k)
		print("new ssk done")
		gram[j][i] = gram[i][j]

#normalize
for i in range(0,len(train_docs)):
	for j in range(0, len(train_docs)):
		gram[i][j] = gram[i][j]/math.sqrt(gram[i][i]*gram[j][j])

Y = np.array(train_labels)
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

model = svm.SVC(kernel='precomputed')
model.fit(gram, Y)

test_gram = np.zeros((len(test_docs),len(train_docs)))
for i in range(0, len(test_docs)):
	for j in range(0, len(train_docs)):
		test_gram[i][j] = ssk.getSSK(test_docs[i],train_docs[j], k)

#normalize
for i in range(0,len(test_docs)):
	for j in range(0, len(test_docs)):
		test_gram[i][j] = test_gram[i][j]/math.sqrt(test_gram[i][i]*test_gram[j][j])

print(test_gram)

predicted = model.predict(test_gram)

Y = np.array(test_labels)
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

countNumberOfRights = 0
for i in range(len(Y)):
	if(predicted[i] == Y[i]):
		countNumberOfRights += 1

print("right:", countNumberOfRights/len(Y))
#model.predict()