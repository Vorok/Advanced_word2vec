import os
import re
# note: multiprocessing is messed up in jupyter
from multiprocessing import Pool
import spacy

# clean up your text and generate list of words for each document. 
nlp = spacy.load('en')
def clean_up(text):  
	removal=['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']
	text_out = []
	doc= nlp(text)
	for token in doc:
		if token.is_stop == False and token.is_alpha and len(token)>2 and token.pos_ not in removal:
			lemma = token.lemma_
			text_out.append(lemma)
	return text_out
	
if __name__ == '__main__':
	file = open("Short_sample.txt", "r")
	doclist = [line for line in file]
	docstr = ''.join(doclist)
	sentences = re.split(r'[.!?]', docstr)
	sentences = [sentence for sentence in sentences if len(sentence) > 1]

	pool = Pool(processes=10)
	sentences_parsed = pool.map(clean_up, sentences)

	with open("Short_sample_cleaned.txt", 'w') as f:
		for sentence in sentences_parsed:
			for word in sentence:
				f.write("%s " % word)
			f.write(".\n")