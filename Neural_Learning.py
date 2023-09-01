import pandas as pd
import numpy as np
from PlatinumGroup1BaseData import textdatabase_df, text_cleansed_df, fullcleanse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pickle

count_vect = CountVectorizer()
count_vect.fit(text_cleansed_df)

X = count_vect.transform(text_cleansed_df)
print("Feature extraction selesai")

#pickle.dump(count_vect, open('featurenl.p,', 'wb'))

classes = textdatabase_df.sentimentlabel01

X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.2)

model = MLPClassifier()
model.fit(X_train, y_train)

print("Model training selesai")
#pickle.dump(model, open('modelnl.p', 'wb'))

test = model.predict(X_test)

print("Testing selesai")

print(classification_report(y_test, test))

print("""
      Dalam tes klasifikasi akurasi model, ditemukan bahwa model memiliki akurasi yang sangat baik dalam menginterpretasikan sentimen positif (0.86 F-1 score)
      Namun nilai F-1 untuk estimasi sentimen negatif tidak sebaik akurasi sestimasi sentimen positif. Adapun akurasi dalam mendeteksi sentimen netral masih dalam kategori passable namun perlu ditingkatkan kedepannya
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

    clf = MLPClassifier()

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

print("Progress aman sampai titik ini...")




