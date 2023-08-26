
from tensorflow.keras.datasets import  mnist 
import matplotlib.pyplot as plt

#%% loading data

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#%% REshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)

print(X_train.shape)
print(X_test.shape)

plt.imshow(X_train[2,:,:])
plt.show()
#%% Normalize data
X_train = X_train/255
X_test = X_test/255
plt.imshow(X_train[2,:,:])
plt.show()

#%% define model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten,Dense

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))


model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#%%
hist = model.fit(X_train, Y_train, epochs=10)

model.evaluate(X_test, Y_test)

#%%
plt.plot(hist.history['accuracy'], label='Training accuracy')
plt.plot(hist.history['loss'], label='Training loss')

plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(loc='center right')
plt.show()