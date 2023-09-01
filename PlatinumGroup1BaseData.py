import pandas as pd
import regex as re
import nltk

textdatabase_df = pd.read_csv('/Users/januardopanggabean/VSCE Platinum Challenge/train_preprocess.tsv.txt', sep='\t', names=['text', 'sentimentlabel01'])

#print(textdatabase_df.isna())

#print(textdatabase_df)

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n','', text)
    text = re.sub('  +',' ', text)
    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub('  +',' ', text) 
    return text

def textcleansing(text):
    lowertext = lowercase(text)
    lowercharfix = remove_unnecessary_char(lowertext)
    lowercharalpha = remove_nonaplhanumeric(lowercharfix)
    return lowercharalpha

textdatabase_df['text-cleansed'] = textdatabase_df['text'].apply(lambda x:textcleansing(x))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwordsindo = nltk.corpus.stopwords.words('indonesian')
def stopwordscleanse(text):
    tokenizedwords= word_tokenize(text)
    filteredtokens = [tokenwords for tokenwords in tokenizedwords if tokenwords.lower() not in stopwordsindo]
    filteredtext = ' '.join(filteredtokens)
    return filteredtext

textdatabase_df['text-token-clean'] = textdatabase_df['text-cleansed'].apply(lambda x:stopwordscleanse(x))
text_cleansed_df = textdatabase_df['text-cleansed'].apply(lambda x:stopwordscleanse(x))

def fullcleanse(text):
    phase1 = textcleansing(text)
    phase2 = stopwordscleanse(phase1)
    return phase2

print('Modul ini telah selesai. Lanjut ke modul selanjutnya')

#textdatabase_df.to_csv('/Users/januardopanggabean/VSCE Platinum Challenge/data/textdatamk2.csv', index=False)
