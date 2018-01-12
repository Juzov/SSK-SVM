import sys
import os
from sklearn import svm
import sklearn.preprocessing as preprocessing
import numpy as np
import math
import re
#import ssk_prune as ssk
from ssk_cache import StringSubsequenceKernel
from ssk_prune import StringSubsequenceKernelWithPrune
import time
import string_functions
from joblib import Parallel, delayed

# SSK needs higher recursion limit, this could be a problem at certain computers
sys.setrecursionlimit(100000)

to_prune = True
#spam or reuters?
is_spam = True

#Setting SSK Parameters
k = 3
lambda_decay = 0.5
m = 7
ssk = None

if(to_prune):
	theta = 3*k
	ssk = StringSubsequenceKernelWithPrune(k,lambda_decay,theta)
else:
	ssk = StringSubsequenceKernel(k,lambda_decay)

test_docs, train_docs, train_labels, test_labels = string_functions.get_info(is_spam)

# Only use 20 documents
test_docs = test_docs[:3]+test_docs[-3:]
train_docs = train_docs[:3]+train_docs[-3:]
test_labels = test_labels[:3]+test_labels[-3:]
train_labels = train_labels[:3]+train_labels[-3:]

gram = np.zeros((len(train_docs),len(train_docs)))

# Get the most frequent subsequences in the spam corpus
most_used = string_functions.get_most_used(is_spam)
print(most_used)
print("Most used done")
start = time.time()

# Approximate gram matrix
cacheForSSK = {}

def inner_loop(i,j):
	global most_used
	global train_docs
	global cacheForSSK

	ij_instance = 0
	for x in range(0, len(most_used)):
		cacheForSSK[(j,most_used[x][0])] = ssk.run_instance(train_docs[j],most_used[x][0])
		ij_instance	+= ssk.run_instance(train_docs[i],most_used[x][0])*cacheForSSK[(j,most_used[x][0])]
	return [[i,j],ij_instance]

gram_array = Parallel(n_jobs=-1)(delayed(inner_loop)(i,j) for i in range(0,len(train_docs)) for j in range(i, len(train_docs)))
print (gram_array)

#for i in range(0,len(train_docs)):
#	for j in range(i, len(train_docs)):
#		gram[i][j] = gram_array.pop(0)
#		gram[j][i] += gram[i][j]

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
		for x in range(0, len(most_used)):
			if ((j,most_used[x][0]) in cacheForSSK):
				test_gram[i][j] = ssk.run_instance(test_docs[i],most_used[x][0])*cacheForSSK[(j,most_used[x][0])]
			else:
				test_gram[i][j] = ssk.run_instance(test_docs[i],most_used[x][0])*ssk.run_instance(train_docs[j],most_used[x][0])
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
