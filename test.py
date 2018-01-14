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
	