from sklearn import svm
import nltk as nltk
from nltk.corpus import reuters
import numpy as np

def ssk_kernel(X, Y):
	#TODO
	"add kernel"
	print(X)
	print(Y)

documents = reuters.fileids()
 
train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))
 
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

train_labels = [reuters.categories(doc_id)
                                  for doc_id in train_docs_id]
test_labels = [reuters.categories(doc_id)
                             for doc_id in test_docs_id]

# TODO:
"Construct the x and y, look at the labels and how they have done it in the report"
X = np.array(train_docs).reshape(1,-1)
Y = np.array(train_labels).reshape(1,-1)

clf = svm.SVC(kernel=ssk_kernel)
clf.fit(X, Y)

#clf.predict()