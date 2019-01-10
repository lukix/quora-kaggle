def removeUnwantedWords(questionsWordsLists):
	wordsToRemove = ['a', 'to', 'of', 'and']
	return map(
		lambda question: filter(
			lambda word: word not in wordsToRemove,
			question,
		),
		questionsWordsLists,
	)