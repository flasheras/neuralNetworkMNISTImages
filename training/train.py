# Libraries
import tensorflow as tf


# Import Fashion MNIST from tensorflow sample dataset.
# fashion_mnist = input_data.read_data_sets('input/data', one_hot=True)
fashion_mnist = tf.keras.datasets.fashion_mnist
# mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# Explore images dataset
print(len(train_images[0]))  # 28x28 Colour value Matrix
print(len(train_images[0][0]))
print(train_images[0])

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = tf.keras.Sequential()
# Add layers. Input layer: size 28x28.
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Output layer 10x1
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
# test with 10,000 images
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('10,000 image Test accuracy:', test_acc)
