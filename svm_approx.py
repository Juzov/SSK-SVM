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
import ssk_old as ssk
import substrings
import time

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

# SSK needs higher recursion limit, this could be a problem at certain computers
sys.setrecursionlimit(100000)
k = 3
m = 7

# Uncomment/comment this for reuters
test_docs, train_docs, train_labels, test_labels = get_reuters()
# Uncomment/comment this for spam
#test_docs, train_docs, train_labels, test_labels = get_spam()

# Only use 20 documents
test_docs = test_docs[:3]+test_docs[-3:]
train_docs = train_docs[:3]+train_docs[-3:]
test_labels = test_labels[:3]+test_labels[-3:]
train_labels = train_labels[:3]+train_labels[-3:]

gram = np.zeros((len(train_docs),len(train_docs)))

# Get the most frequent subsequences in the spam corpus
mostUsed = substrings.getMostUsed()
print(mostUsed)
print("Most used done")
start = time.time()

# Approximate gram matrix
cacheForSSK = {}
for i in range(0,len(train_docs)):
	for j in range(i, len(train_docs)):
		for x in range(0, len(mostUsed)):
			cacheForSSK[(j,mostUsed[x][0])] = ssk.getSSK(train_docs[j],mostUsed[x][0], k)
			gram[i][j] += ssk.getSSK(train_docs[i],mostUsed[x][0], k)*cacheForSSK[(j,mostUsed[x][0])]
			gram[j][i] += gram[i][j]
	print("Document", i+1,"/",len(train_docs),"done")

# Normalize gram matrix
for i in range(0,len(train_docs)):
	for j in range(0, len(train_docs)):
		gram[i][j] = gram[i][j]/math.sqrt(gram[i][i]*gram[j][j])

# Format the training labels
Y = np.array(train_labels).reshape(-1)
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

# Train SVM
model = svm.SVC(kernel='precomputed')
model.fit(gram, Y)
print("Training done")

# Approximate training gram matrix
test_gram = np.zeros((len(test_docs),len(train_docs)))
for i in range(0, len(test_docs)):
	for j in range(0, len(train_docs)):
		for x in range(0, len(mostUsed)):
			if ((j,mostUsed[x][0]) in cacheForSSK):
				test_gram[i][j] = ssk.getSSK(test_docs[i],mostUsed[x][0], k)*cacheForSSK[(j,mostUsed[x][0])]
			else:
				test_gram[i][j] = ssk.getSSK(test_docs[i],mostUsed[x][0], k)*ssk.getSSK(train_docs[j],mostUsed[x][0], k)
	print("Test document", i+1,"/",len(test_docs),"done")

# Normalize training gram matrix
for i in range(0,len(test_gram)):
	for j in range(0, len(test_gram[0])):
		test_gram[i][j] = test_gram[i][j]/math.sqrt(test_gram[i][i]*test_gram[j][j])

# Test the model
print("Predicting...")
predicted = model.predict(test_gram)

stop = time.time()
# Format Y labels
Y = np.array(test_labels).reshape(-1)
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)


print("Time:", stop-start)
# Calculate error rate
countNumberOfRights = 0
print(predicted)
print(Y)
for i in range(len(Y)):
	if(predicted[i] == Y[i]):
		countNumberOfRights += 1

print("right:", countNumberOfRights/len(Y))
#model.predict()
