from tqdm import tqdm
import operator

tqdm.pandas()

def checkCoverage(vocab, embeddingsIndex):
	a = {}
	oov = {}
	k = 0
	i = 0
	for word in tqdm(vocab):
		try:
			a[word] = embeddingsIndex[word]
			k += vocab[word]
		except:
			oov[word] = vocab[word]
			i += vocab[word]
			pass

	print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
	print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
	sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

	return sorted_x