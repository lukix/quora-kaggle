from tqdm import tqdm

tqdm.pandas()

def buildVocabulary(sentences, verbose =  True):
	"""
	:param sentences: list of list of words
	:return: dictionary of words and their count
	"""
	vocab = {}
	for sentence in tqdm(sentences, disable = (not verbose)):
		for word in sentence:
			try:
				vocab[word] += 1
			except KeyError:
				vocab[word] = 1
	return vocab