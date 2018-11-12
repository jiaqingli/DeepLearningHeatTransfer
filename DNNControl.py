import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


#read data
num = 3100
training_num = 3000
testing_num = 3000 + 1
rand_index = np.random.randint(0, 60000, num)
data = np.load('data/training_data.npz')
imgs = data['input_images'][rand_index]
targets = data['training_targets'][rand_index]
#shuffle data
#np.random.shuffle(imgs)
#targets[imgs[:, 0]]

targets_one_max = np.amax(targets[:, 0])
targets_one_min = np.amin(targets[:, 0])
targets_two_max = np.amax(targets[:, 1])
targets_two_min = np.amin(targets[:, 1])
targets_three_max = np.amax(targets[:, 2])
targets_three_min = np.amin(targets[:, 2])
targets_four_max = np.amax(targets[:, 3])
targets_four_min = np.amin(targets[:, 3])
targets_five_max = np.amax(targets[:, 4])
targets_five_min = np.amin(targets[:, 4])

targets[:,0] = (2*(targets[:,0] - targets_one_min)/(targets_one_max - targets_one_min)) - 1
targets[:,1] = (2*(targets[:,1] - targets_two_min)/(targets_two_max - targets_two_min)) - 1
targets[:,2] = (2*(targets[:,2] - targets_three_min)/(targets_three_max - targets_three_min)) - 1
targets[:,3] = (2*(targets[:,3] - targets_four_min)/(targets_four_max - targets_four_min)) - 1
targets[:,4] = (2*(targets[:,4] - targets_five_min)/(targets_five_max - targets_five_min)) - 1



#build model
img_shape = imgs[0].shape
model = Sequential()

model.add(Conv2D(8, (3, 3), input_shape=img_shape))
model.add(Activation('relu'))
model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(5))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

#train
model.fit(imgs[0:training_num], targets[0:training_num], batch_size=64, epochs=2)
model.summary()

#evaluate
model.evaluate(imgs[testing_num:], targets[testing_num:], verbose=True)
