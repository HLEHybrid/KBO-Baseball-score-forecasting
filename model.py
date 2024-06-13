import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from pickle import dump
from imblearn.over_sampling import ADASYN

if __name__ == "__main__":

    data = pd.read_excel('gamelog_agg.xlsx')

    x = data.iloc[:, 2:-1].values
    y = data.iloc[:, -1:].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = tf.keras.utils.to_categorical(y)
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("클래스 매핑:", class_mapping)

    x_train, x_test_have_num, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = x_train[:,4:]
    x_test = x_test_have_num[:,4:]

    # RandomOverSampler 객체 생성
    adasyn = ADASYN(random_state=42)
    x_train, y_train = adasyn.fit_resample(x_train, y_train)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    # StandardScaler 객체 생성
    scaler = StandardScaler()
    scaler.fit_transform(x_train)

    # scaler 저장하기
    dump(scaler,open('scaler.pkl','wb'))

    # #scaler 불러오기
    # from pickle import load
    # load_scaler = load(open('scaler.pkl','rb'))

    # x 데이터 스케일링
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)


    filepath = 'models/kbo_model.{epoch:02d}.hdf5'
    modelckpt = ModelCheckpoint(filepath=filepath)

    # Sequential 모델 생성
    model = Sequential()

    # Input layer와 hidden layer 추가
    model.add(Dense(units=128, activation='relu', input_shape=(np.shape(x_train)[1],)))
    model.add(Dropout(0.03))
    # 추가적인 hidden layer와 output layer 추가
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val),callbacks=[modelckpt])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('loss learning_curve.png')

    loss = history.history['accuracy']
    val_loss = history.history['val_accuracy']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_loss, 'b', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('accuracy learning_curve.png')