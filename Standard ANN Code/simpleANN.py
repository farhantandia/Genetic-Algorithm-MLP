import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adam,SGD,Adadelta
import pickle
from keras.layers.advanced_activations import LeakyReLU,ReLU,Softmax
import time
import logging

'''''
##############set logging configurations####################
'''''
logger=logging.getLogger(__name__)

logger.setLevel(level=logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')
file_Handler = logging.FileHandler(filename='SimpleANN.log')
file_Handler.setFormatter(formatter)

logger.addHandler(file_Handler)

'''
##################################################################
'''

f = open("dataset_features_10.pkl", "rb")
data_inputs = pickle.load(f)
f.close()

f = open("outputs_10.pkl", "rb")
data_outputs = pickle.load(f)
f.close()
print(data_inputs.shape)

X_train, X_test, y_train, y_test = train_test_split(data_inputs, data_outputs, test_size=0.25)

batch_size = 64
num_classes = 10
epochs = 100

logger.info('batch_size: {} num_classes: {} epochs: {} '.format(batch_size, num_classes, epochs))
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(150,input_shape=(360,)))

model.add(ReLU())
# model.add(Dropout(0.2))
model.add(Dense(256))
model.add(ReLU())
model.add(Dense(512))
model.add(ReLU())
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()
start_time = time.time() 
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

elapsed_time = time.time() - start_time  

logger.info('Elapsed time :{} Test loss: {} Test accuracy: {}'.format(elapsed_time, score[0], score[1]))
print('Elapsed time for processing in second: ',elapsed_time)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


history.history['val_acc'].insert(0, 0)
plt.plot(history.history['acc'])
plt.title('ANN Optimizer performance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0,1)
plt.savefig('results.png')
plt.show()
