import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

# Generate random images for training data.
images = np.random.uniform(size=(100, 8, 8, 3)).astype(np.float32)
n, width, height, channels = images.shape

# Reshape images to achieve desired autoregressivity.
event_shape = [height * width * channels]
reshaped_images = tf.reshape(images, [n, -1])

  # Density estimation with MADE.
# made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape,hidden_units=[20, 20], activation='elu')
# distribution = tfd.TransformedDistribution(
#     distribution=tfd.Normal(loc=0., scale=1.),
#     bijector=tfb.MaskedAutoregressiveFlow(
#       lambda x: tf.unstack(made(x), num=2, axis=-1)),
#     event_shape=event_shape)

# maf =
distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
            hidden_layers=[512, 512], activation=tf.nn.elu)),
    event_shape=event_shape)

# Construct and fit model.
x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)

model.compile(optimizer=tf.optimizers.Adam(),
            loss=lambda _, log_prob: -log_prob)

batch_size = 10
model.fit(x=reshaped_images,
        y=np.zeros((n, 0), dtype=np.float32),
        batch_size=batch_size,
        epochs=10,
        steps_per_epoch=n // batch_size,
        shuffle=True,
        verbose=True)

# Use the fitted distribution.
s = distribution.sample((3, 1))
distribution.log_prob(np.ones((5, 8, 8, 3), dtype=np.float32))
distribution.log_prob(s)