#!/usr/bin/python3

import os
import textmining
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

DRAKE = 'drake'
KENDRICK = 'kendrick'
DATASET_FOLDER = 'dataset'

def get_dataframe(indir):
	rows = []
	index = []
	for root, dirs, filenames in os.walk(os.path.join(DATASET_FOLDER, indir)):
		for each_file in filenames:
			with open(os.path.join(DATASET_FOLDER, indir, each_file)) as f:
				rows.append({'body': f.read().replace('\n', ' '), 'class': indir})
				index.append(each_file.split('.')[0])
	data_frame = pd.DataFrame(rows, index=index)
	return data_frame

def main():
	drake_dataframe = get_dataframe("drake")
	kendrick_dataframe = get_dataframe("kendrick_lamar")
	dataset = drake_dataframe.append(kendrick_dataframe)
	dataset = dataset.reindex(np.random.permutation(dataset.index))

	count_vectorizer = CountVectorizer()
	counts = count_vectorizer.fit_transform(dataset['body'].values)

	examples = ["america"]
	pipeline = Pipeline([
	    ('vectorizer',  CountVectorizer()),
	    ('classifier',  MultinomialNB()) ])

	pipeline.fit(dataset['body'].values, dataset['class'].values)
	predictions = pipeline.predict(examples)
	print(predictions)

if __name__ == "__main__":
	main()
