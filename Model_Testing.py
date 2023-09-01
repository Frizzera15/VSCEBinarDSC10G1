import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from PlatinumGroup1BaseData import text_cleansed_df, textdatabase_df, fullcleanse
import pickle

# Preview Text Yang Akan Diproses
print(text_cleansed_df.head())
print(text_cleansed_df.shape)

count_vect = CountVectorizer()
count_vect.fit(text_cleansed_df)

X = count_vect.transform(text_cleansed_df)

pickle.dump(count_vect, open("feature.p", 'wb'))
print("Feature Extraction Selesai Disini")

# Referensi nama kolom adalah sentimentlabel01 
classes = textdatabase_df.sentimentlabel01
print(classes)

X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)

pickle.dump(model, open("model.p", 'wb'))
print('Training Selesai')

test = model.predict(X_test)
print("Testing selesai dilakukan")

print(classification_report(y_test, test))
print("""
      Sejauh ini akurasi penentuan sentimen positif yang terbaik akurasinya karena mendekati 1 (0.89). 
      Untuk model estimasi sentimen netral dan negatif sudah cukup namun perlu ditingkatkan lagi dengan data tambahan.
      """)

# Cross Validation model lintas partisi 1 hingga 5

kf = KFold(n_splits=5,random_state=42, shuffle=True)
accuracies = []
y = classes

for iteration, data in enumerate(kf.split(X), start=1):

    data_train = X[data[0]]
    target_train = y[data[0]]

    data_test = X[data[1]]
    target_test = y[data[1]]

    clf = MultinomialNB()

    clf.fit(data_train, target_train)

    preds = clf.predict(data_test)

    accuracy = accuracy_score(target_test, preds)

    print('Training/Iterasi ke =', iteration)
    print(classification_report(target_test, preds))
    print("============================================================")

    accuracies.append(accuracy)

avg_accuracy = np.mean(accuracies)

print()
print()
print()
print("Rata-rata accuracy adalah", avg_accuracy)

print("""
      Sejauh ini melihat hasil dari Cross Validation lintas 5 partisi data yang digunakan, angka rata rata akurasi yang mencapai 0.839 menunjukkan bahwa konsistensi model sudah cukup baik dalam menentukan sentimen atas teks yang diajukan.
      Namun masih terdapat ruang peningkatan kualitas untuk deteksi sentimen netral dan negatif dikarenakan seiring jumlah iterasi model dilakukan, akurasi F-1 untuk model sentimen netral dan negatif belum pernah melewati angka 0.8
      """ )

original_text = "saya pergi ke kantor untuk bekerja"


text = count_vect.transform([fullcleanse(original_text)])

result = model.predict(text)[0]

print("Sentiment:")
print(result)


print('Safe Progress')

