"""
Implementing logistic regression using TensorFlow

Liu, Yuxi (Hayden). Python Machine Learning By Example: Build intelligent systems using Python, TensorFlow 2, PyTorch, and scikit-learn, 3rd Edition (p. 178). Packt Publishing. Kindle Edition. 
"""

import tensorflow as tf
import pandas as pd
n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train].astype('float32')
X_test = X[n_train:]
Y_test = Y[n_train:].astype('float32')

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train).toarray().astype('float32')
X_test_enc = enc.transform(X_test).toarray().astype('float32')

# we use the tf.data api to shuffle & batch data

batch_size = 1000
train_data = tf.data.Dataset.from_tensor_slices((X_train_enc, Y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# we define the weights and bias of the logistic regression model:

n_features = int(X_train_enc.shape[1])
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))

# we then create a gradient descent optimizer that searches for the best coef by minimizing the loss
# we herein use Adam as our optimizer

learning_rate = 0.0008
optimizer = tf.optimizers.Adam(learning_rate)

# we define optimization process where we compute the current prediction and cost and
# update the model coef following the computed gradients

def run_optimization(x, y):
    with tf.GradientTape() as g:
        logits = tf.add(tf.matmul(x, W), b)[:, 0]
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits))
    gradients = g.gradient(cost, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

# we run the traininng for 6k steps (one step is with one batch of random samples):

training_steps = 6000
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % 500 == 0:
        logits = tf.add(tf.matmul(batch_x, W), b)[:, 0]
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = batch_y, logits = logits))
        print("step: %i, loss: %f" % (step, loss))

# after the model is trained, we use it to make predictions on the testing set and report the AUC metric:

logits = tf.add(tf.matmul(X_test_enc, W), b)[:, 0]
pred = tf.nn.sigmoid(logits)
auc_metric = tf.keras.metrics.AUC()
auc_metric.update_state(Y_test, pred)

print(f'AUC on testing set: {auc_metric.result().numpy():.3f}')

