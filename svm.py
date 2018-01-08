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

def format_text(text):
	#document = (reuters.raw(documentIDList[0]))
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	text.lower()
	textWihoutStopWords = pattern.sub('', text)
	textWihoutSymbols = re.sub(r'[^a-zA-Z\d\s]','', textWihoutStopWords)
	formattedText = re.sub(r"\s+", " ", textWihoutSymbols)
	return formattedText

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

def ssk_kernel(X, Y):
	#TODO
	"add kernel"
	print(X)
	print(Y)

sys.setrecursionlimit(100000)
k = 10

#test_docs, train_docs, train_labels, test_labels = get_reuters()
test_docs, train_docs, train_labels, test_labels = get_spam()
test_docs = test_docs[:5]+test_docs[697:]
train_docs = train_docs[:5]+train_docs[697:]
test_labels = test_labels[:5]+test_labels[697:]
train_labels = train_labels[:5]+train_labels[697:]

gram = np.zeros((len(train_docs),len(train_docs)))
for i in range(0,len(train_docs)):
	for j in range(i, len(train_docs)):
		gram[i][j] = ssk.getSSK(train_docs[i],train_docs[j], k)
		print("new ssk done")
		gram[j][i] = gram[i][j]

#normalize
for i in range(0,len(train_docs)):
	for j in range(0, len(train_docs)):
		gram[i][j] = gram[i][j]/math.sqrt(gram[i][i]/gram[j][j])

# TODO:
"Construct the x and y, look at the labels and how they have done it in the report"
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
		test_gram[i][j] = test_gram[i][j]/math.sqrt(test_gram[i][i]/test_gram[j][j])

predicted = model.predict(test_gram)

Y = np.array(test_labels)
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

print(sum(predicted == Y)/len(Y))

#model.predict()