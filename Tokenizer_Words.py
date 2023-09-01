import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('tokenization')
from nltk.tokenize import word_tokenize
stopwordsindo = nltk.corpus.stopwords.words('indonesian')
print(stopwordsindo[:10])

##EXAMPLES
sentence = 'Saya butuh uang cepat hari ini'
tokenizedwords = word_tokenize(sentence)
print(tokenizedwords)