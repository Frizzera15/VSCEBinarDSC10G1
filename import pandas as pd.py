import pandas as pd

df = pd.read_csv('/Users/januardopanggabean/VSCE Platinum Challenge/train_preprocess.tsv.txt', sep='\t', names=['text','sentiment_labels'])
print(df)

