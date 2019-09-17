import os
import json
import pprint
import datetime
from pylab import *
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from time import time
import sklearn.decomposition
import scipy.stats
import scipy
from optim.lr_scheduler import *
import random
import glob

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def load_dataset(args):

    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.train == 'train':
        fnames_data = [r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/train-images-idx3-ubyte',
                   r'C:\Users\justjo\Downloads\public_datasets/FasionMNIST/t10k-images-idx3-ubyte']
        fnames_test = [r'C:\Users\justjo\Downloads\public_datasets/MNIST/train-images.idx3-ubyte',
                   r'C:\Users\justjo\Downloads\public_datasets/MNIST/t10k-images.idx3-ubyte']
        data_train = read_idx(fnames_data[0])
        data_train = data_train.reshape((data_train.shape[0], -1))/128 - 1

        data_val = read_idx(fnames_data[1])
        data_val = data_val.reshape((data_val.shape[0], -1))/128 - 1
        data_test = np.concatenate([read_idx(fnames_test[0]), read_idx(fnames_test[1])])
        data_test = data_test.reshape((data_test.shape[0], -1)) / 128 - 1

        if args.svd:
            _, _, vh = scipy.linalg.svd(data_train, full_matrices=False)
            data_train = np.matmul(data_train, vh.T)[:,:args.n_comp_pca]
            data_val = np.matmul(data_val, vh.T)[:,:args.n_comp_pca]
            data_test = np.matmul(data_test, vh.T)[:,:args.n_comp_pca]
        ### _,_,vh = scipy.linalg.svd()
    elif args.train == 'cifar':
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
        if args.svd:
            _, _, vh = scipy.linalg.svd(cifardata, full_matrices=False)
            data_train = np.matmul(cifardata, vh.T)[:, :args.n_comp_pca]
            data_val = np.matmul(cifar_val_data, vh.T)[:, :args.n_comp_pca]
            data_test = np.matmul(svhndata, vh.T)[:, :args.n_comp_pca]
        else:
            data_train = cifardata
            data_val = cifar_val_data
            data_test = svhndata
    else:
        fnames_data = ['data/MNIST/train-images.idx3-ubyte', 'data/MNIST/t10k-images.idx3-ubyte', 'data/FasionMNIST/train-images-idx3-ubyte', 'data/FasionMNIST/t10k-images-idx3-ubyte']
        data = []
        for f in fnames_data:
            data.append(read_idx(f))
        data = np.concatenate(data)
        data = data.reshape((data.shape[0], -1))/128 - 1
        data_train = data
        data_val = []
        data_test = []

    dataset_train = tf.data.Dataset.from_tensor_slices(tf.constant(data_train, dtype=tf.float32))#.float().to(args.device)
    # dataset_train = dataset_train.shuffle(buffer_size=data_train.shape[0]).map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    dataset_train = dataset_train.shuffle(buffer_size=data_train.shape[0]).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices(tf.constant(data_val, dtype=tf.float32))#.float().to(args.device)
    # dataset_valid = dataset_valid.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    dataset_valid = dataset_valid.batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices(tf.constant(data_test, dtype=tf.float32))#.float().to(args.device)
    dataset_test = dataset_test.batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    args.n_dims = data_train.shape[1]
    return dataset_train, dataset_valid, dataset_test

class MAF(tf.keras.models.Model):

    def __init__(self, *, output_dim, num_layers, num_flows, num_hidden_dim, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_flows = num_flows
        self.num_hidden_dim = num_hidden_dim
        self.num_layers = num_layers
        self.shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(hidden_layers=[self.num_hidden_dim]*self.num_layers, activation=tf.nn.elu) #hidden_layers default = [512, 512]
        # self.shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(hidden_layers=[3072, 3072]) #hidden_layers default = [512, 512]
        # Defining the bijector
        num_bijectors = num_flows #5
        # num_bijectors = 5
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

@tf.function #Adding the tf.function makes it about 10 times faster!!!
def train_step(model, X, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = -tf.reduce_mean(model.flow.log_prob(X)) ## model caches information from sample to calc prob
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

class parser_:
    pass

# tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.dataset = 'cifarSVHN'  # 'gq_ms_wheat_johnson'#'gq_ms_wheat_johnson' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
args.batch_dim = 10
args.epochs = 5000
args.patience = 10
args.cooldown = 10
args.decay = 0.5
args.min_lr = 5e-4
args.flows = 3
args.layers = 1
args.tensorboard = 'tensorboard'
args.hidden_dim = 3072*2
args.load = ''#r'checkpoint/corn_layers1_h12_flows6_gated_2019-09-07-12-08-42'
args.save = True
args.early_stopping = 10
args.manualSeed = None
args.manualSeedw = None
args.prefetch_size = 1  # data pipeline prefetch buffer size
args.parallel = 16  # data pipeline parallel processes
args.train = 'cifar'  # 'train', 'pca', 'cifar
args.n_comp_pca = 3072 #784 = full mnist/Fmnist #3072 = full cifar10/svhn
args.svd = False

args.path = os.path.join('checkpoint', '{}{}_layers{}_h{}_flows{}{}_{}'.format(
    args.n_comp_pca, args.dataset, args.layers, args.hidden_dim, args.flows, 'SVD' if args.svd else '',
    str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)

if args.save and not args.load:
    print('Creating directory experiment..')
    os.mkdir(args.path)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

# X = (np.random.randn(10, 500)).astype(np.float32)
# print(X.shape)
print('Creating model..')
with tf.device(args.device):
    model = MAF(output_dim=args.n_dims, num_layers=args.layers, num_flows=args.flows, num_hidden_dim = args.hidden_dim)
# model.summary() #Yields an error. The model needs called before it is build.
x_ = tf.keras.layers.Input(shape=[args.n_dims], dtype=tf.float32)
log_prob_ = model.flow.log_prob(x_)
model_ = tf.keras.Model(x_, log_prob_)
model.compile(optimizer=tf.optimizers.Adam(),
            loss=lambda _, log_prob: -log_prob)

model.fit(x=np.concatenate([x for x in data_loader_train]),
        y=np.zeros((50000, 0), dtype=np.float32),
        epochs=10,
           shuffle=True,
           verbose=True)
        #    ,
        # validation_data=np.concatenate([x for x in data_loader_valid]))

# X = [x for x in data_loader_train][0]
# _ = model.flow.bijector.inverse(X)
# X=[]
model.summary()

writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
writer.set_as_default()

tf.compat.v1.train.get_or_create_global_step()

global_step = tf.compat.v1.train.get_global_step()
global_step.assign(0)

root = None

print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.keras.optimizers.Adam()

root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.compat.v1.train.get_global_step())

if args.load:
    load_model(args, root, load_start_epoch=True)
print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

# Training
with tf.device(args.device):
    start = time()
    for i in range(args.epochs):
        train_loss = []
        for x_mb in data_loader_train:
            loss = train_step(model, x_mb, optimizer)
            train_loss.append(loss)
            # if (i % save_interval == 0):
            #     print(i, " ",loss.numpy(), (time()-start))
            #     start = time()
        train_loss = np.mean(train_loss)
        validation_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)/args.n_dims) for x_mb in data_loader_valid])
        test_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb) / args.n_dims) for x_mb in data_loader_test])
        stop = scheduler.on_epoch_end(epoch = epoch, monitor=validation_loss)
        tf.compat.v1.train.get_global_step().assign_add(1)
        tf.summary.scalar('loss/validation', validation_loss,tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/test', test_loss, tf.compat.v1.train.get_global_step())
        if stop:
            break

    validation_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)/args.n_dims) for x_mb in data_loader_valid])
    test_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)/args.n_dims) for x_mb in data_loader_test])

    # validation_loss = - np.median([np.median(compute_log_p_x(model, x_mb)) for x_mb in data_loader_valid])
    # test_loss = - np.median([np.median(compute_log_p_x(model, x_mb)) for x_mb in data_loader_test])

    print('###### Stop training after {} epochs!'.format(epoch + 1))
    print('Validation loss: {:4.3f}'.format(validation_loss))
    print('Test loss:       {:4.3f}'.format(test_loss))

# Sampling from the trained model
# XF = model.flow.sample(10000)
# plt.scatter(XF[:, 0], XF[:, 1], s=5, color='blue')