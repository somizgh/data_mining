
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import Imputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

def preprocessing(data, label_name, drop_col =[]):

    datadrop = data.drop(drop_col, axis=1)
    datadrop = datadrop.dropna(axis=0)

    objects = datadrop.select_dtypes(include='object')
    numeric = datadrop.drop(objects.columns, axis=1)
    dum = pd.get_dummies(objects, drop_first=True)
    datax = pd.concat([numeric, dum], axis=1)

    min_max_scaler = MinMaxScaler()
    fitted = min_max_scaler.fit(datax)
    output = min_max_scaler.transform(datax)

    output = pd.DataFrame(output, columns=datax.columns, index=list(datax.index.values))

    label = output[label_name]
    dum_label = pd.get_dummies(label)
    datax = output.drop([label_name], axis=1)

    return datax,dum_label
'''
pd.set_option('display.max_row',500)
pd.set_option('display.max_columns', 100)
data = pd.read_csv('./titanic/train.csv')


output,label = preprocessing(data, 'Survived', ['PassengerId', 'Name', 'Ticket', 'Cabin','SibSp','Parch'])
X_train, X_test, Y_train, Y_test = train_test_split(output, label, test_size=0.33, random_state=42)

print("X_train",X_train.info())
'''

train_data = pd.read_csv("train_data.csv",index_col=0)
test_data = pd.read_csv("test_data.csv",index_col=0)
target_data = pd.read_csv("target.csv",index_col=0)
print(train_data.shape)


print(train_data.head())
print(test_data.head())
print(target_data.head())
print(train_data.info())


def create_model():
    model = Sequential()

    model.add(Dense(20, input_shape=(8,)))
    model.add(Activation("relu"))

    model.add(Dense(14))
    model.add(Activation("relu"))


    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.0001), metrics=["accuracy"])
    return model

#model.add(Dense(200))
#model.add(Activation("relu"))




cb_checkpoint = ModelCheckpoint(filepath="./learn_character_model", monitor='val_loss',
                                verbose=1, save_best_only=True)

cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model = KerasClassifier(build_fn=create_model, epochs=400, batch_size=10, verbose=0)

k_fold = KFold(n_splits=4, shuffle=True, random_state=0)

results = cross_val_score(model,train_data,target_data,cv=k_fold)
print(results)
'''

hist = model.fit(train_data.values, target_data.values, batch_size=32, epochs=1000, validation_data=(X_test.values, Y_test.values), verbose=1, callbacks=[cb_checkpoint, cb_early_stopping])

loss, accuracy = model.evaluate(X_test,Y_test, batch_size = 64, verbose = 1)
print("Accuracy = {}%, ".format(accuracy*100))




fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
'''