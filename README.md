# Work Package WP2 : Artificial / Convolutional Neural Network 


# Import TensorFlow library
import tensorflow as tf

# Load the dataset
dataset = tf.keras.datasets.mnist

# Split the dataset into training and test sets
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc

please continue in full

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Save the model
model.save('tf_model.h5')

print('Model saved!')

finish

# Print the results
print('Test accuracy:', test_acc)"
please now integrate all 3 into 1 
.

# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Feature Digital Concept Model
# Input from the designer
x1 = np.array([20, 50, 70, 90, 110, 130, 150, 170, 190])
y1 = np.array([30, 80, 120, 150, 190, 220, 250, 280, 310])

# Create the TensorFlow graph
X = tf.placeholder(tf.float32)
Y = tf.




# Create the TensorFlow graph
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Model parameters
W = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')

# Linear model
y_predict = W*X + b

# Loss
loss = tf.reduce_mean(tf.square(y_predict - Y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Run
with tf.Session() as sess:
    sess.run(init)
    
    # Train
    for epoch in range(1000):
        sess.run(train, {X: x1, Y: y1})
        
    # Get the optimized parameter
    final_w, final_b = sess.run([W, b])"

# Plot the result
x1_test = np.linspace(20, 190, 10)
y1_test = final_w*x1_test + final_b

plt.plot(x1_test, y1_test, 'r-', label='2D Sketch Model')
plt.plot(x1, y1, 'bo', label='Designer Input')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Transform model into 3D
x2 = np.array([20, 50, 70, 90, 110, 130, 150, 170, 190])
y2 = np.array([30, 80, 120, 150, 190, 220, 250, 280, 310])
z1 = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])

# Create the TensorFlow graph
X3 = tf.placeholder(tf.float32)
Y3 = tf.placeholder(tf.float32)
Z3 = tf.placeholder(tf.float32
and continue

# Create the TensorFlow graph
X3 = tf.placeholder(tf.float32)
Y3 = tf.placeholder(tf.float32)
Z3 = tf.placeholder(tf.float32)

# Model parameters
Wz = tf.Variable(0.0, name='weight_z')
bz = tf.Variable(0.0, name='bias_z')

# Linear model
z_predict = Wz*X3 + bz

# Loss
loss3 = tf.reduce_mean(tf.square(z_predict - Z3))

# Optimizer
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train3 = optimizer3.minimize(loss3)

# Initialize variables
init3 = tf.global_variables_initializer()

# Run
with tf.Session() as sess:
    sess.run(init3)
    
    # Train
    for epoch in range(1000):
        sess.run(train3, {X3: x

continue

# Run
with tf.Session() as sess:
    sess.run(init3)
    
    # Train
    for epoch in range(1000):
        sess.run(train3, {X3: x2, Y3: y2, Z3: z1})
        
    # Get the optimized parameter
    final_wz, final_bz = sess.run([Wz, bz])

# Plot the result
x1_test3 = np.linspace(20, 190, 10)
y1_test3 = final_wz*x1_test3 + final_bz
z1_test3 = final_wz*x1_test3 + final_bz

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x2, y2, z1, c="b", marker="o", label="Designer Input")
ax.scatter(x1_test3, y1_test3, z1_test3, c="r", marker="o", label="

please continue

# Plot the result
x1_test3 = np.linspace(20, 190, 10)
y1_test3 = final_wz*x1_test3 + final_bz
z1_test3 = final_wz*x1_test3 + final_bz

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x2, y2, z1, c="b", marker="o", label="Designer Input")
ax.scatter(x1_test3, y1_test3, z1_test3, c="r", marker="o", label="3d Sketch Model")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Load the dataset
dataset = tf.keras.datasets.mnist

# Split the dataset into training and test sets
(train_images, train_labels), (test_images, test_labels) = dataset.load_

continue please

# Split the dataset into training and test sets
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Save the model
model.save

is that it

# Save the model
model.save('tf_model.h5')

print('Model saved!')

# Print the results
print('Test accuracy:', test_acc)

finish

print("TensorFlow algorithm is successfully executed and a model is produced from the given dataset!")

how can you execute this to develop and run on CNN?

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape the data
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model 
please continue

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Save the model
model.save('tf_cnn_model.h5')

print('Model saved!')

# Print the results
print('Test accuracy:', test_acc)

print("TensorFlow CNN algorithm is successfully executed and a model is produced from the given dataset!")
