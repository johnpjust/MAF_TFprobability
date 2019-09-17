import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
import glob
import scipy
import scipy.stats
from scipy import io

## data
fnames_cifar = glob.glob(r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\train\*')
cifar10 = [np.load(f, allow_pickle=True, encoding='latin1') for f in fnames_cifar]
cifardata = np.concatenate([a['data'] for a in cifar10])
# cifarlabels = np.expand_dims(np.concatenate([a['labels'] for a in cifar10]), axis=1)
cifardata = cifardata / 128 - 1

cifar_val = np.load(r'C:\Users\justjo\Downloads\public_datasets\cifar-10-python\cifar-10-batches-py\test_batch', allow_pickle=True, encoding='latin1')
cifar_val_data = cifar_val['data']/128 - 1

svhn = scipy.io.loadmat(r'C:\Users\justjo\Downloads\public_datasets\SVHN.mat')
svhndata = np.moveaxis(svhn['X'], 3, 0)
svhndata = np.reshape(svhndata, (svhndata.shape[0], -1))
svhndata = svhndata / 128 - 1

_, _, vh = scipy.linalg.svd(cifardata, full_matrices=False)
cifardata = np.matmul(cifardata, vh.T)#[:, :args.n_comp_pca]
cifar_val_data = np.matmul(cifar_val_data, vh.T)#[:, :args.n_comp_pca]
svhndata = np.matmul(svhndata, vh.T)#[:, :args.n_comp_pca]

# Generate random images for training data.
# images = np.random.uniform(size=(100, 8, 8, 3)).astype(np.float32)
# n, width, height, channels = images.shape
images = cifardata
n, event_shape = images.shape
event_shape = [event_shape]

# Reshape images to achieve desired autoregressivity.
# event_shape = [height * width * channels]
# reshaped_images = tf.reshape(images, [n, -1])
reshaped_images = images

  # Density estimation with MADE.
# made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape,hidden_units=[20, 20], activation='elu')
# distribution = tfd.TransformedDistribution(
#     distribution=tfd.Normal(loc=0., scale=1.),
#     bijector=tfb.MaskedAutoregressiveFlow(
#       lambda x: tf.unstack(made(x), num=2, axis=-1)),
#     event_shape=event_shape)

shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(hidden_layers=[3072] * 1, activation=tf.nn.tanh)  # hidden_layers default = [512, 512]
# self.shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(hidden_layers=[3072, 3072]) #hidden_layers default = [512, 512]
# Defining the bijector
num_bijectors = 3
# num_bijectors = 5
bijectors = []
for i in range(num_bijectors):
    bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=shift_and_log_scale_fn))
    # bijectors.append(tfb.Permute(permutation=[1, 0]))
    bijectors.append(tfp.bijectors.Permute(np.random.permutation(event_shape[0])))
# Discard the last Permute layer.
bijector = tfb.Chain(list(reversed(bijectors[:-1])))

# Defining the flow
distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    # distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
    bijector=bijector,
    event_shape=event_shape)


# distribution = tfd.TransformedDistribution(
#     distribution=tfd.Normal(loc=0., scale=1.),
#     bijector=tfb.MaskedAutoregressiveFlow(
#         shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
#             hidden_layers=[3072, 3072], activation=tf.nn.tanh)),
#     event_shape=event_shape)

# Construct and fit model.
x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)
model.summary()

model.compile(optimizer=tf.optimizers.Adam(),
            loss=lambda _, log_prob: -log_prob*0.00001)

batch_size = 100
model.fit(x=reshaped_images,
        y=np.zeros((n, 0), dtype=np.float32),
        batch_size=batch_size,
        epochs=100,
        steps_per_epoch=n // batch_size,
        shuffle=True,
        verbose=True,
          validation_data = [cifar_val_data.astype(np.float32), np.zeros((10000, 0), dtype=np.float32)],
          callbacks = [tfk.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='auto', baseline=None, restore_best_weights=True)])

temp=distribution.bijector.inverse(cifardata.astype(np.float32)[:1000,:])

NLLtrain = np.concatenate([model(x) for x in np.array_split(cifardata.astype(np.float32), 50)])
NLLval = np.concatenate([model(x) for x in np.array_split(cifar_val_data.astype(np.float32), 50)])
NLLtest = np.concatenate([model(x) for x in np.array_split(svhndata.astype(np.float32), 50)])

# Use the fitted distribution.
# s = distribution.sample((3, 1))
# distribution.log_prob(np.ones((5, 8, 8, 3), dtype=np.float32))
# distribution.log_prob(s)

import matplotlib.pyplot as plt
plt.scatter(temp[:,0], temp[:,1])
dist = scipy.stats.johnsonsu.fit(NLLtrain)
temp1=np.arcsinh((NLLtrain - dist[-2]) / dist[-1]) * dist[1] + dist[0]
temp2=np.arcsinh((NLLval - dist[-2]) / dist[-1]) * dist[1] + dist[0]
temp3=np.arcsinh((NLLtest - dist[-2]) / dist[-1]) * dist[1] + dist[0]
plt.figure();plt.hist(temp3, 50, density=True, alpha=0.5, label='svhn');plt.hist(temp1, 50, density=True, alpha=0.5, label='cifar_train');plt.hist(temp2, 50, density=True, alpha=0.5, label='cifar_val')
plt.legend()
plt.xlabel('MAF Density')
plt.savefig(r'C:\Users\justjo\Desktop\myfigure.png', bbox_inches='tight')