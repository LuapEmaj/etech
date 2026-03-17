import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

# ================== DATA PREPARATION STAGE ================== #

# 1.) Load training and testing images in "coarse" labels
(coarse_train, coarse_TrLabels), (coarse_test, coarse_TsLabels) = keras.datasets.cifar100.load_data(label_mode='coarse')
print('Coarse Class: {}'.format(np.unique(coarse_TrLabels)))

# 2.) Load training and testing images in "fine" labels
(fine_train, fine_Trlabels), (fine_test, fine_TsLabels) = keras.datasets.cifar100.load_data(label_mode='fine')
print('Fine Class for all: {}'.format(np.unique(fine_Trlabels)))

# 3.) Extract all images of a specific coarse class from TRAINING DATASET
# "4" = fruits and vegetables
idx = []
for i in range(len(coarse_TrLabels)):
    if coarse_TrLabels[i] == 4:
        idx.append(i)
print('Total images with coarse label 4 (fruits and vegetables) from TRAINING DATASET: {}'.format(len(idx)))
idx = np.array(idx)

# 4.) Extract images and corresponding fine labels
train_images, train_labels = fine_train[idx], fine_Trlabels[idx]
print("Shape of the training dataset: {}".format(train_images.shape))
uniq_fineClass = np.unique(train_labels)
print('Fine classes for the extracted training images: {}'.format(uniq_fineClass))

# 5.) Extract all images of coarse class 4 from TESTING DATASET
idx = []
for i in range(len(coarse_TsLabels)):
    if coarse_TsLabels[i] == 4:
        idx.append(i)
print('Total images with coarse label 4 (fruits and vegetables) from TESTING DATASET: {}'.format(len(idx)))
idx = np.array(idx)

# 6.) Extract images and corresponding fine labels
test_images, test_labels = fine_test[idx], fine_TsLabels[idx]
print("Shape of the testing dataset: {}".format(test_images.shape))
uniq_fineClass = np.unique(test_labels)
print('Fine classes for the extracted testing images: {}'.format(uniq_fineClass))

# 7.) Relabel training and testing datasets to start from zero (0)
for i in range(len(uniq_fineClass)):
    for j in range(len(train_labels)):
        if train_labels[j] == uniq_fineClass[i]:
            train_labels[j] = i
    for j in range(len(test_labels)):
        if test_labels[j] == uniq_fineClass[i]:
            test_labels[j] = i

# 8.) Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# 9.) Plot a few samples from the TESTING DATASET
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(test_labels[i])
plt.show()

# ================== BUILD AND TRAIN CNN MODEL ================== #

model = tf.keras.Sequential()

# 32 convolution filters, each 3x3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

# Max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add dropout after pooling to prevent overfitting (randomly turns off 25% of neurons)
model.add(Dropout(0.25))

# Flatten to feed into fully connected layer
model.add(Flatten())

# Fully connected layer - reduced from 128 to 64 to have fewer parameters and prevent overfitting
model.add(Dense(64, activation='relu'))

# Add dropout before output layer (randomly turns off 50% of neurons)
model.add(Dropout(0.5))

# Output layer
model.add(Dense(len(uniq_fineClass), activation='softmax'))

# Summary
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Early stopping callback - stops training when validation loss stops improving (prevents overfitting)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model with more epochs (15) but early stopping will stop if validation loss stops improving
history = model.fit(
    train_images, 
    train_labels, 
    epochs=15,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# ================== PLOT TRAINING & VALIDATION LOSS ================== #
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.clf()
plt.plot(epochs, loss, 'g-', label="Training loss")
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ================== TEST THE MODEL ================== #

# Define string classes manually for your coarse class "fruits and vegetables"
str_class = ['apple', 'mushroom', 'orange', 'pear', 'bell pepper']

print(test_images.shape)
print("Classes in the testing images: {}".format(np.unique(test_labels)))

# Evaluate using model.evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Total number of testing images: {}'.format(len(test_images)))
print('Test accuracy:', test_acc)

# Predict using model.predict
classification = model.predict(test_images)
print('\nPrediction of the first test input image:\n{}'.format(classification[0]))

# Index of maximum probability
max_prob_idx = np.argmax(classification[0])
print('Predicted class: {} -- {}'.format(max_prob_idx, str_class[max_prob_idx]))

# True class
idx = test_labels[0]
print('True class: {} -- {}'.format(idx, str_class[idx[0]] if isinstance(idx, np.ndarray) else str_class[idx]))