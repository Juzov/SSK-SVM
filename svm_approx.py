import sys
import os
from sklearn import svm
import sklearn.preprocessing as preprocessing
import numpy as np
import math
import re
#import ssk_prune as ssk
import time
import string_functions
from joblib import Parallel, delayed

# SSK needs higher recursion limit, this could be a problem at certain computers
sys.setrecursionlimit(100000)

def inner_loop(i,j, most_used, train_docs, cacheForSSK, ssk):
		ij_instance = 0
		for x in range(0, len(most_used)):
			ij_instance	+= ssk.run_instance(train_docs[i],most_used[x][0])*cacheForSSK[(j,x)]
		return ij_instance

def inner_loop_testing(i,j,most_used, test_docs, cacheForSSK, ssk):
	ij_instance = 0
	for x in range(0, len(most_used)):
		ij_instance	+= ssk.run_instance(test_docs[i],most_used[x][0])*cacheForSSK[(j,x)]
	return ij_instance

def inner_loop_cache(j,x,most_used,train_docs, ssk):
	result = ssk.run_instance(train_docs[j],most_used[x][0])
	return [j, x, result]

#where k also advocates the length of most used words e.g only words of size 5
def svm_calc(is_spam,amount_of_documents, ssk, word_amount, k):
	#print(most_used)
	print("Most used done")
	start = time.time()

	test_docs, train_docs, train_labels, test_labels = string_functions.get_info(is_spam, amount_of_documents)
	most_used = string_functions.get_most_used(word_amount,k,train_docs)

	gram = np.zeros((len(train_docs),len(train_docs)))

	# Approximate gram matrix
	cacheForSSK = {}

	cache_array = Parallel(n_jobs=-1)(delayed(inner_loop_cache)(i,x,most_used,train_docs,ssk) for i in range(0,len(train_docs)) for x in range(0, len(most_used)))

	for i in range(0,len(cache_array)):
		cacheForSSK[(cache_array[i][0],cache_array[i][1])] = cache_array[i][2]

	gram_array = Parallel(n_jobs=-1)(delayed(inner_loop)(i,j,most_used,train_docs,cacheForSSK,ssk) for i in range(0,len(train_docs)) for j in range(i, len(train_docs)))

	for i in range(0,len(train_docs)):
		for j in range(i, len(train_docs)):
			gram[i][j] = gram_array.pop(0)
			gram[j][i] = gram[i][j]

	#print(gram)

	# Normalize gram matrix
	unnormalizedTrain = np.zeros(len(train_docs))
	for i in range(0,len(train_docs)):
		# if(math.isnan(gram[i][i]) or math.isnan(gram[j][j]) or gram[i][i] == 0 or gram[j][j] == 0):
		# 	unnormalizedTrain[i] = 0
		# else:
		# 	unnormalizedTrain[i] = gram[i][i]
		for j in range(0, len(train_docs)):
			if(math.isnan(gram[i][i]) or math.isnan(gram[j][j]) or gram[i][i] == 0 or gram[j][j] == 0):
				gram[i][j] = 0
			else:
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

	test_gram_array = Parallel(n_jobs=-1)(delayed(inner_loop_testing)(i,j,most_used,test_docs,cacheForSSK,ssk) for i in range(0,len(test_docs)) for j in range(0, len(train_docs)))
	for i in range(0,len(test_gram)):
		for j in range(0, len(test_gram[0])):
			test_gram[i][j] = test_gram_array.pop(0)

	# for i in range(0, len(test_docs)):
	# 	for j in range(0, len(train_docs)):
	# 		for x in range(0, len(most_used)):
	# 			if ((j,most_used[x][0]) in cacheForSSK):
	# 				test_gram[i][j] += ssk.run_instance(test_docs[i],most_used[x][0])*cacheForSSK[(j,most_used[x][0])]
	# 			else:
	# 				test_gram[i][j] += ssk.run_instance(test_docs[i],most_used[x][0])*ssk.run_instance(train_docs[j],most_used[x][0])
	# 	print("Test document", i+1,"/",len(test_docs),"done")

	# Normalize training gram matrix
	for i in range(0,len(test_gram)):
		for j in range(0, len(test_gram[0])):
			if(math.isnan(test_gram[i][i]) or test_gram[i][i] == 0 or math.isnan(test_gram[j][j]) or test_gram[j][j] == 0):
				test_gram[i][j] = 0
			else:
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

	elapsed_time = stop-start

	print("Time:", elapsed_time)
	# Calculate error rate
	countNumberOfRights = 0
	print(predicted)
	print(Y)
	for i in range(len(Y)):
		if(predicted[i] == Y[i]):
			countNumberOfRights += 1

	accuracy = countNumberOfRights/len(Y)
	print("right:", accuracy)
	
	return accuracy, elapsed_time
