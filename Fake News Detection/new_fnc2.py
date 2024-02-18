# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset= pd.read_csv('train_stances.csv')
dataset1=pd.read_csv('train_bodies.csv')
result=pd.merge(dataset,dataset1)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 49972):
    body = re.sub('[^a-zA-Z]', ' ', result['articleBody'][i])
    body = body.lower()
    body = body.split()
    ps = PorterStemmer()
    body = [ps.stem(word) for word in body if not word in set(stopwords.words('english'))]
    body = ' '.join(body)
    corpus.append(body)
'''count_vec = CountVectorizer(stop_words="english", analyzer='word', 
                            ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=4)
count_train = count_vec.fit(corpus)
bag_of_words = count_vec.transform(corpus)
print(count_vec.get_feature_names())'''


X = dataset.iloc[:, 0].values
df1 = pd.DataFrame(X)
y = dataset.iloc[:, 2].values
df2 = pd.DataFrame(y)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
df2 = labelencoder_y.fit_transform(df2)


'''from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
txt_fitted = tf.fit(corpus)
txt_transformed = txt_fitted.transform(corpus) 
print(tf.vocabulary_)
idf = tf.idf_
print(dict(zip(txt_fitted.get_feature_names(), idf)))'''

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).todense()

#vocab size
X.shape

#check vocabulary using below command
print(cv.vocabulary_)

#get feature names
print(cv.get_feature_names())

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
txt_fitted = tf.fit_transform(corpus)
'''txt_transformed = txt_fitted.transform(corpus) 
print(tf.vocabulary_)
idf = tf.idf_
print(dict(zip(txt_fitted.get_feature_names(), idf)))'''

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(txt_fitted)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = tf.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = tf.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = tf.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)'''