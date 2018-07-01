import gzip
import pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print(train_y[57])


# TODO: the neural net!!

train_y = one_hot(train_y, 10)  # the labels are in the last row. Then we encode them in one hot code

x = tf.placeholder("float", [None, 784])  # samples 28*28 píxeles correspondiente a las imaágenes
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 40)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(40)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(40, 20)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W3 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b3 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
i = tf.nn.sigmoid(tf.matmul(h, W2) + b2)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(i, W3) + b3)
#y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y)) # Error que se le pasa al optimizador

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20
list_error_training = []
list_error_validation = []
epoca = 0
error = -1

while epoca < 100:
    for jj in range((int)(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error_training = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    error_validation = sess.run(loss, feed_dict={x: valid_x, y_: one_hot(valid_y, 10)})

    list_error_training.append(error_training)
    list_error_validation.append(error_validation)

    print("Epoch #:", epoca, "Error: ", error_training)
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")
    epoca = epoca + 1
    if(len(list_error_validation) >= 2 and abs(error_validation - error) < 0.5):
        break

print("----------------------")
print("   Testing...  ")
print("----------------------")

n_accert = 0
y_test = one_hot(test_y, 10)
dataset_testing = len(y_test)
result_testing = sess.run(y, feed_dict={x: test_x})

for estimate, real in zip(y_test, result_testing):
    if np.argmax(estimate) == np.argmax(real):
        n_accert = n_accert + 1

capacidad_predictiva = (n_accert / dataset_testing)*100
print("La capacidad predictiva de la red neuronal es: " + str(capacidad_predictiva) + "%")

x_error_train = list(range(1, len(list_error_training) + 1))

plt.plot(x_error_train, list_error_training)
plt.title("Gráfica Conjunto de Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.show()

plt.plot(x_error_train, list_error_validation)
plt.title("Gráfica Conjunto de Validación")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.show()