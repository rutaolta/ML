import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def convert_to_sparse_matrix(docs):
	indptr = [0]
	indices = []
	data = []
	vocabulary = {}
	for d in docs:
		for term in d:
			index = vocabulary.setdefault(term, len(vocabulary))
			indices.append(index)
			data.append(1)
		indptr.append(len(indices))
	return csr_matrix((data, indices, indptr), dtype=int)


def get_data():
	data_dir = 'messages'
	messages = []
	y = []
	for filename in os.listdir(data_dir):
		with open(os.path.join(data_dir, filename)) as file:
			subject = file.readline()
			skip = file.readline()
			content = file.readline()
			messages.append(content.split())
		if "legit" in filename:
			y.append("legit")
		else:
			y.append("spam")

	X = convert_to_sparse_matrix(messages)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	return X_train, X_test, y_train, y_test


# print(X_train, X_test, y_train, y_test)