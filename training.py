#%%
import tensorflow as tf
import numpy as np
from tensorflow_graphics.geometry.transformation import quaternion as quat
import matplotlib.pyplot as plt

#%%
# Load data
joints_file = open('joints_quat_values.npy', 'rb')
ee_file = open('ee_quat_poses.npy', 'rb')
joints_data = np.load(joints_file)
ee_data = np.load(ee_file)
joints_file.close()
ee_file.close()

# %% check the data 
print(joints_data[0])
print(ee_data[0])

# %% split training data
dataset = tf.data.Dataset.from_tensor_slices((ee_data, joints_data)).prefetch(1)
DATASET_SIZE = len(dataset)
train_size = int(0.8*DATASET_SIZE)
val_size = int(0.1*DATASET_SIZE)
test_size = int(0.1*DATASET_SIZE)

print(f'DATASET_SIZE={DATASET_SIZE}')
print(list(dataset.take(1).as_numpy_iterator()))
print(train_size, val_size, test_size)
# %%
train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size).take(val_size)
test_ds = dataset.skip(train_size).skip(val_size)

print(len(train_ds), len(val_ds), len(test_ds))

# %% Create batches for training
BATCH_SIZE = 8

# flatten the target_y results
train_batches = train_ds.shuffle(train_size//4).map(lambda x,y: (x,tf.reshape(y, [-1]))).batch(BATCH_SIZE).prefetch(1)
train_val = val_ds.map(lambda x,y: (x,tf.reshape(y, [-1]))).batch(BATCH_SIZE).prefetch(1) # does not shuffling since we do not backpropagate here


# %% check shape of batches
print(train_batches.as_numpy_iterator().next()[0].shape) # input
print(train_batches.as_numpy_iterator().next()[1].shape) # target
print(train_batches.as_numpy_iterator().next()[1]) # target sample

# %% Create model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(7,), batch_size=BATCH_SIZE),
        tf.keras.layers.Dense(256, activation='relu', batch_size=BATCH_SIZE),
        tf.keras.layers.Dense(164, activation='relu'),
        tf.keras.layers.Dense(16, name='joints_values')
    ])
    return model

model = build_model()
model.summary()

# %% Training
epochs=24
init_lr = 1e-4
optimizer = tf.keras.optimizers.Adam(lr=init_lr)
# optimizer = tf.keras.optimizers.SGD(lr=init_lr, momentum=0.9)

model.compile(
    optimizer = optimizer,
    loss = tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError()])


history = model.fit(train_batches, validation_data=train_val, epochs=epochs)


# %% Plotting

# metric_loss = history.history['mean_squared_error']
# val_metric_loss = history.history['val_mean_squared_error']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='loss')
plt.plot(epochs_range, val_loss, label='val_loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, metric_loss, label='mean_squared_error')
# plt.plot(epochs_range, val_metric_loss, label='val_mean_squared_error')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Accuracy')

plt.show()

# %% Evaluate model
ds = test_ds.map(lambda x,y: (x,tf.reshape(y, [-1]))).batch(BATCH_SIZE).prefetch(1)
results = model.evaluate(ds, batch_size=8)
model.predict(ds.take(1).take(1))
# %%
