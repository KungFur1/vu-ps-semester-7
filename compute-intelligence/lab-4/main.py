import os
import random
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import visualize
import metrics
import time


def get_num_classes(path="data"):
    image_classes = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    return len(image_classes)


def get_classes(path="data"):
    return [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]


def index_data(path="data"):
    image_classes = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    X = []
    Y = []
    for image_class in image_classes:
        class_path = os.path.join(path, image_class)
        file_list = [os.path.join(class_path, file) for file in os.listdir(class_path)]
        X += file_list
        Y += [image_class for i in range(len(file_list))]
    return X, Y


def shuffle_data(X, Y):
    data = list(zip(X, Y))
    random.shuffle(data)
    X, Y = zip(*data)
    return list(X), list(Y)


def read_and_preprocess_image(path):
    image = mpimg.imread(path)
    image = resize(image, (128, 128), anti_aliasing=True)
    image = image / 255.0 if image.max() > 1 else image
    return image


def one_hot_encode(Y, path="data"):
    image_classes = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    Y_encoded = []
    for label in Y:
        new_entry = [0] * len(image_classes)
        new_entry[image_classes.index(label)] = 1
        Y_encoded.append(new_entry)
    return Y_encoded


def split_data(data, train_size=0.8, val_size=0.1):
    if(train_size + val_size > 1 ):
        raise Exception()
    
    train_index = int(len(data) * train_size)
    val_index = train_index + int(len(data) * val_size)
    
    data_train = data[:train_index]
    data_val = data[train_index:val_index]
    data_test = data[val_index:]

    return numpy.array(data_train), numpy.array(data_val), numpy.array(data_test)


def load_images(path_list):
    return [read_and_preprocess_image(path) for path in path_list]


X, Y = index_data()
X, Y = shuffle_data(X, Y)
Y = one_hot_encode(Y)

X = load_images(X)

X_train, X_val, X_test = split_data(X)
Y_train, Y_val, Y_test = split_data(Y)


# model = Sequential([
#     Input(shape=(128, 128, 3)),
#     Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(get_num_classes(), activation='softmax')
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model = Sequential([
#     Input(shape=(128, 128, 3)),
#     Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(get_num_classes(), activation='softmax')
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model = Sequential([
#     Input(shape=(128, 128, 3)),
#     Conv2D(32, kernel_size=(3, 3), activation='linear'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),

#     Conv2D(64, kernel_size=(3, 3), activation='linear'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(128, kernel_size=(3, 3), activation='linear'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.3),
    
#     Flatten(),
#     Dense(128, activation='linear'),
#     Dropout(0.4),
#     Dense(5, activation='softmax')
# ])
# model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])


model = tf.keras.models.load_model("model-0.keras")
# history = model.fit(X_train, Y_train, epochs=12, batch_size=32, validation_data=(X_val, Y_val))
# model.save("model-0.keras")


# visualize.plot_training_history(history)
metrics.evaluate_and_plot_confusion_matrix(model, X_test, Y_test, get_classes())



# Make Predictions!
predictions = model.predict(X_test)
predicted_classes = numpy.argmax(predictions, axis=1)
predicted_labels = [get_classes()[i] for i in predicted_classes]

Y_index = numpy.argmax(Y_test, axis=1)
Y_labels = [get_classes()[i] for i in Y_index]

print(Y_labels[:30])
print(predicted_labels[:30])
