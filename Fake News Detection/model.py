

from getEmbeddings import getEmbeddings,getEmbeddings2
from untitled1 import make_feature_vector
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scikitplot.plotters as skplt
import pandas as pd
import os

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte, ypred)
    plt.show()

# Read the data
if not os.path.isfile('./xtr.npy') or \
    not os.path.isfile('./xte.npy') or \
    not os.path.isfile('./ytr.npy') or \
    not os.path.isfile('./yte.npy'):
        xtr,xte,ytr,yte = getEmbeddings()
        np.save('./xtr', xtr)
        np.save('./xte', xte)
        np.save('./ytr', ytr)
        np.save('./yte', yte)

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')


def baseline_model():
    '''Neural network with 3 hidden layers'''
    
    model = Sequential()
    model.add(Dense(200, input_dim=300, activation='relu', kernel_initializer='normal'))

    model.add(Dense(200, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', kernel_initializer='normal'))
    model.add(Dense(2, activation="softmax", kernel_initializer='normal'))

    # gradient descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    # configure the learning process of the model
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

# Train the model
model = baseline_model()
print(model.summary())
x_train, x_test, y_train, y_test = train_test_split(xtr, ytr, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y = np_utils.to_categorical((label_encoder.transform(y_train)))
label_encoder.fit(y_test)
encoded_y_test = np_utils.to_categorical((label_encoder.transform(y_test)))
estimator = model.fit(x_train, encoded_y, epochs=20, batch_size=64)
print("Model Trained!")
score = model.evaluate(x_test, encoded_y_test)
print("")
print("Accuracy = " + format(score[1]*100, '.2f') + "%")  

probabs = model.predict_proba(x_test)
y_pred = np.argmax(probabs, axis=1)
 
# Draw the confusion matrix
plot_cmat(y_test, y_pred)

model.save("model with acc 0.92"+"_recall_"+".h5")

df=pd.read_csv(r"C:\Users\hp\Downloads\Machine_Learning_AZ_Template_Folder\Machine Learning A-Z Template Folder\minor\live2.csv")
print("df",df.shape)

#text=make_feature_vector(r"C:\Users\hp\Downloads\Machine_Learning_AZ_Template_Folder\Machine Learning A-Z Template Folder\minor\live2.csv",300)
#print("text",text.shape)
#print(text)


text= getEmbeddings2()

pred=model.predict_proba(text)
print(pred)
result = np.argmax(pred, axis=1)
print(result)