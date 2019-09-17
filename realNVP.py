from pylab import *
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from time import time

class MAF(tf.keras.models.Model):

    def __init__(self, *, output_dim, num_masked, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked
        self.shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(hidden_layers=[128, 128], activation=tf.nn.elu)
        # Defining the bijector
        num_bijectors = 5
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.shift_and_log_scale_fn))
            # bijectors.append(tfb.Permute(permutation=[1, 0]))
            bijectors.append(tfp.bijectors.Permute(np.random.permutation(self.output_dim)))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0]*self.output_dim),
            # distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    def getFlow(self, num):
        return self.flow.sample(num)


X = (np.random.randn(100, 3) * np.array([1, 3, 5]) + np.array([-3, -10, 4])).astype(np.float32)
print(X.shape)


model = MAF(output_dim=3, num_masked=1)
# model.summary() #Yields an error. The model needs called before it is build.
_ = model(X)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function #Adding the tf.function makes it about 10 times faster!!!
def train_step(X):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = -tf.reduce_mean(model.flow.log_prob(X))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training
start = time()
for i in range(1001):
    loss = train_step(X)
    if (i % 50 == 0):
        print(i, " ",loss.numpy(), (time()-start))
        start = time()

# Sampling from the trained model
XF = model.flow.sample(10000)
plt.scatter(XF[:, 0], XF[:, 1], s=5, color='blue')