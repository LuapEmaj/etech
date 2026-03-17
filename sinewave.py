import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import numpy as np
import math
import matplotlib.pyplot as plt

# ================== SETTINGS ================== #
SAMPLES = 1000
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================== GENERATE SINE WAVE ================== #
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
y_values = np.sin(x_values)

# Add noise
y_noise = y_values.copy()
y_noise += 0.1 * np.random.randn(*y_values.shape)

# ================== SPLIT DATA ================== #
TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_noise, [TRAIN_SPLIT, TEST_SPLIT])
y_clean = y_values[TEST_SPLIT:]

# ================== PLOT DATA ================== #
plt.plot(x_values, y_values, 'b.', label='Original sine wave')
plt.plot(x_train, y_train, 'g.', label='Training noisy samples')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave with Training Samples')
plt.legend()
plt.show()

# ================== CHECK SHAPES ================== #
print('x_train shape: {}'.format(x_train.shape))

# ================== BUILD MODEL ================== #
model = tf.keras.Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.summary()

# ================== TRAIN MODEL ================== #
metricInfo = model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_validate, y_validate)
)

# ================== PLOT TRAINING & VALIDATION LOSS ================== #
loss = metricInfo.history['loss']
val_loss = metricInfo.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ================== PREDICTIONS & VISUALIZATION ================== #
predictions = model.predict(x_test)

# Sort test data for smooth plot
sorted_idx = np.argsort(x_test)
x_test_sorted = x_test[sorted_idx]
y_clean_sorted = y_clean[sorted_idx]
predictions_sorted = predictions[sorted_idx]

plt.clf()
plt.plot(x_test_sorted, y_clean_sorted, 'b.', label='Observation')
plt.plot(x_test_sorted, predictions_sorted, 'r*', label='Predictions')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Predictions vs Observations on Test Data')
plt.legend()
plt.show()

# ================== SAVE MODEL ================== #
model.save('baseModel.h5')
print('Model saved as baseModel.h5')