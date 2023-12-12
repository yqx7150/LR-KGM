# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np
import os

class GetMRI(Dataset):
    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])
        self.augment = None

    def k2wgt(self,X,W):
        result = np.multiply(X,W) 
        return result
    def im2row(self,im,winSize):
        size = (im).shape
        out = np.zeros(((size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),winSize[0]*winSize[1],size[2]),dtype=np.complex64)
        count = -1
        for y in range(winSize[1]):
            for x in range(winSize[0]):
                count = count + 1                 
                temp1 = im[x:(size[0]-winSize[0]+x+1),y:(size[1]-winSize[1]+y+1),:]
                temp2 = np.reshape(temp1,[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')
                out[:,count,:] = np.squeeze(temp2) # MATLAB reshape          
                
        return out

    def __getitem__(self,index):
        siat_input=loadmat(self.data_names[index])['Img2']
        siat=np.array(siat_input[:,:,0:2],dtype=np.float32)
  
        siat_complex = siat[:,:,0]+1j*siat[:,:,1]
        siat_kdata = np.fft.fftshift(np.fft.fft2(siat_complex))

        weight=loadmat('../input_data/weight1.mat')['weight']
        ksize=[8,8]
        kdata_w = self.k2wgt(siat_kdata,weight) 

        kdata=np.stack((kdata_w,kdata_w,kdata_w,kdata_w,kdata_w,kdata_w,kdata_w,kdata_w),2)
        # 256*256*8
        # print("kkkkkkkkkkkkkdata.shape: ")
        # print(kdata.shape)
        # zl hankel=self.im2row(kdata,ksize)
        
        # zl size_temp = hankel.shape
        # zl A = np.reshape(hankel,[size_temp[0],size_temp[1]*size_temp[2]],order = 'F') # max: 14.925017 (matlab:14.9250) (64009, 128)
        # zl A_temp = np.zeros((121,512,512))
        # ans_1 = np.array(A_temp,dtype=np.complex64)
        # for i in range(121):#diu 49
        #     cut=A[512*i:512*(i+1)]
        #     ans_1[i,:,:]=cut
        
        '''
        for i in range(125):
            plt.ion()
            plt.imshow(np.log(1+abs(A[i])),cmap='gray')
            plt.pause(0.3)
        assert 0
        '''
        ans=np.concatenate((np.real(kdata),np.imag(kdata)),2)

        # ans=ans.transpose(1,2,0)
        ans=ans.transpose((2,0,1))


        return ans
    
    def __len__(self):
        return len(self.data_names)
    
    # def __len__(self):
    #     return len(self.data_names)

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    #dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  #elif config.data.dataset == 'LSUN':
  #  dataset_builder = tfds.builder(f'lsun/{config.data.category}')
  #  train_split_name = 'train'
  #  eval_split_name = 'validation'

  #  if config.data.image_size == 128:
  #    def resize_op(img):
  #      img = tf.image.convert_image_dtype(img, tf.float32)
  #      img = resize_small(img, config.data.image_size)
  #      img = central_crop(img, config.data.image_size)
  #      return img

  #  else:
  #    def resize_op(img):
  #      img = crop_resize(img, config.data.image_size)
  #      img = tf.image.convert_image_dtype(img, tf.float32)
  #      return img
        
  elif config.data.dataset == 'LSUN':
    #dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = crop_resize(img, config.data.image_size)
      img = tf.image.convert_image_dtype(img, tf.float32)
      return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  #train_ds = create_dataset(dataset_builder, train_split_name)
  #eval_ds = create_dataset(dataset_builder, eval_split_name)

  dataset = GetMRI(root= "/zl/dataset_th/siat_256size_train")
  test_dataset = GetMRI(root= "/zl/dataset_th//siat_256size_eval")
  
  
  train_ds = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                                num_workers=4)

  eval_ds = DataLoader(test_dataset, batch_size=config.eval.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)
  
  
  
  
  return train_ds, eval_ds #, dataset_builder
