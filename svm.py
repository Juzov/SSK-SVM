from sklearn import svm
import sklearn.preprocessing as preprocessing
import nltk as nltk
from nltk.corpus import reuters
import numpy as np
import ssk

def ssk_kernel(X, Y):
	#TODO
	"add kernel"
	print(X)
	print(Y)

documents = reuters.fileids()
 
train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))[:10]
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))

train_docs_id = train_docs_id[:10]
test_docs_id = test_docs_id[:10]
 
train_docs = [reuters.raw(doc_id)[:10] for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id)[:10] for doc_id in test_docs_id]

train_labels = [reuters.categories(doc_id)
                                  for doc_id in train_docs_id]
test_labels = [reuters.categories(doc_id)
                             for doc_id in test_docs_id]

gram = np.zeros((len(train_docs),len(train_docs)))

for i in range(0,len(train_docs)):
	for j in range(0, len(train_docs)):
		gram[i][j] = ssk.kN(train_docs[i],train_docs[j],4)

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
		test_gram[i][j] = ssk.kN(test_docs[i],train_docs[i], 4)

predicted = model.predict(test_gram)

Y = np.array(test_labels)
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

print(np.mean(predicted == Y))

#model.predict()