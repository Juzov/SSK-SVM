import os
import nltk as nltk
from nltk.corpus import stopwords
import re

"Remove stopwords and symbols"
def format_text(text):
	pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	text.lower()
	textWihoutStopWords = pattern.sub('', text)
	textWihoutSymbols = re.sub(r'[^a-zA-Z\d\s]','', textWihoutStopWords)
	formattedText = re.sub(r"\s+", " ", textWihoutSymbols)
	return formattedText

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
			test_data[i] = format_text(email.read())
		if("spmsgc" in filename):
			test_labels[i] = 1
		else:
			test_labels[i] = 0

	for i, filename in enumerate(os.listdir(path_train)):
		with open(path_train+filename,'r') as email:
			train_data[i] = format_text(email.read())
		if("spmsgc" in filename):
			train_labels[i] = 1
		else:
			train_labels[i] = 0

	return test_data, train_data, train_labels, test_labels

def getMostUsed(train_data = ''):
	d = {}
	string_length = 5
	test_data, train_data, train_labels, test_labels = get_spam()

	for i, text in enumerate(train_data):
		for j in range(0, len(text)-string_length):
			if text[j:j+string_length] in d:
				d[text[j:j+string_length]] += 1
			else:
				d[text[j:j+string_length]] = 1

	mostUsed = sorted(d.items(), key=lambda x: x[1])
	mostUsed = mostUsed[len(mostUsed)-200:len(mostUsed)]

	return mostUsed
