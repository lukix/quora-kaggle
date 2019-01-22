def removeUnwantedWords(questionsWordsLists):
    wordsToRemove = ['a', 'to', 'of', 'and']
    return list(map(
        lambda question: list(filter(
            lambda word: word not in wordsToRemove,
            question,
        )),
        questionsWordsLists,
    ))
