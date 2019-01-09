def cleanText(x):
	x = str(x)
	for punct in "/-'":
		x = x.replace(punct, ' ')
	for punct in '&':
		x = x.replace(punct, f' {punct} ')
	for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
		x = x.replace(punct, '')
	return x