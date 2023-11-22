# -*- coding: utf-8 -*-
"""
github -> https://github.com/john-fante
kaggle -> https://www.kaggle.com/banddaniel
"""

from IPython.display import clear_output
!pip install catboost
!pip install visualkeras
clear_output()

# Importing dependencies

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import cv2
import visualkeras

from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Dense, Input, BatchNormalization
from tensorflow.keras.layers import Layer, Activation, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score

BATCH_SIZE = 8

with open('/content/drive/MyDrive/Colab Notebooks/CHEMBL286/train_val_test_dict.json', 'r') as f:
  data = json.load(f)

# Creating a function that used converting json to dataframe

def create_df(json_data, main_path, set_name = 'training'):
  images = []
  labels = []

  for i in range(len(json_data[set_name])):
    img_link = main_path + data[set_name][i][0] + '.png'
    img_label = data[set_name][i][1]

    images.append(img_link)
    labels.append(img_label)

  df = pd.DataFrame({'images': images, 'labels' : labels })

  return df

train_data = create_df(data,'/content/drive/MyDrive/Colab Notebooks/CHEMBL286/imgs/', 'training' )
validation_data = create_df(data,'/content/drive/MyDrive/Colab Notebooks/CHEMBL286/imgs/', 'validation' )
test_data = create_df(data,'/content/drive/MyDrive/Colab Notebooks/CHEMBL286/imgs/', 'test' )

print("train images: ", train_data.shape[0])
print("val images: ", validation_data.shape[0])
print("test images: ", test_data.shape[0])

"""#Creating Datasets"""

def img_preprocessing(image, label):
  img = tf.io.read_file(image)
  img = tf.io.decode_png(img, channels = 3)
  img = tf.image.resize(img, size = (150, 150))
  img = tf.cast(img, tf.float32) / 255.0

  return img, label

# Creating dataset loaders

train_loader = tf.data.Dataset.from_tensor_slices(( train_data['images'], train_data['labels'] ))
train_dataset = (train_loader
                 .map(img_preprocessing)
                 .batch(BATCH_SIZE)
                 .shuffle(train_data['images'].shape[0])
                 .prefetch(BATCH_SIZE))

# train dataset for the second model without shuffle
train_dataset_without_shuffle = (train_loader
                                 .map(img_preprocessing)
                                 .batch(BATCH_SIZE)
                                 .prefetch(BATCH_SIZE))



validation_loader = tf.data.Dataset.from_tensor_slices(( validation_data['images'], validation_data['labels'] ))
validation_dataset = (validation_loader
                      .map(img_preprocessing)
                      .batch(BATCH_SIZE)
                      .prefetch(BATCH_SIZE))


test_loader = tf.data.Dataset.from_tensor_slices(( test_data['images'], test_data['labels'] ))
test_dataset = (test_loader
                 .map(img_preprocessing)
                 .batch(BATCH_SIZE)
                 .prefetch(BATCH_SIZE))

# Convolution block class

class ConvBlock(Layer):

  def __init__(self, filters, kernel_size, activation = 'relu', batchnormalization = False, **kwargs):
    super(ConvBlock, self).__init__(**kwargs)

    self.filters = filters
    self.kernel_size = kernel_size
    self.activation = activation
    self.batchnormalization = batchnormalization

    self.conv = Conv2D(filters, kernel_size, padding = 'same')
    self.batch = BatchNormalization()
    self.act = Activation(activation)
    self.pool = MaxPooling2D()


  def call(self, inputs):

    X = self.conv(inputs)
    if self.batchnormalization:
      X = self.batch(X)
      X = self.act(X)
      X = self.pool(X)
      return X
    else:
      X = self.act(X)
      X = self.pool(X)
      return X


  def get_config():
    base_config = super().get_config()

    return {
        **base_config,
        "filters" : self.filters,
        "kernel_size": self.kernel_size,
        "activation" : self.act
    }

# Custom callback for predicting 5 samples from validation dataset during training

# Red color title for a false prediction
# Green color title for a true prediction

class PredictDuringTraining(Callback):
  def on_epoch_end(self, epochs, logs = None):
    samples = list(validation_dataset.take(-1))

    idxs = np.random.permutation(validation_data.shape[0])[:5]
    batch_idx = idxs // BATCH_SIZE
    image_idx = idxs-batch_idx * BATCH_SIZE
    idx = idxs


    p1 = samples[batch_idx[0]][0][image_idx[0]][np.newaxis,...]
    p2 = samples[batch_idx[1]][0][image_idx[1]][np.newaxis,...]
    p3 = samples[batch_idx[2]][0][image_idx[2]][np.newaxis,...]
    p4 = samples[batch_idx[3]][0][image_idx[3]][np.newaxis,...]
    p5 = samples[batch_idx[4]][0][image_idx[4]][np.newaxis,...]


    images = tf.concat([ p1, p2, p3, p4, p5 ], axis = 0 )


    pred = self.model.predict(images, verbose = 0)
    pred = np.squeeze(pred)
    pred = np.round(pred)


    fig, axs = plt.subplots(1,5, figsize = (6,2), dpi = 150)

    #fig.suptitle('Epoch no: ' + str(epochs + 1) + ' ,(github.com/john-fante)', fontsize= 7 )


    for i in range(5):
      axs[i].imshow(samples[batch_idx[i]][0][image_idx[i]])
      axs[i].axis('off')

      if (samples[batch_idx[i]][1][image_idx[i]].numpy() == pred[i]):
        if (pred[i] == 0):
          axs[i].set_title('Non active', fontsize = 8, color = 'green' )
        elif (pred[i] == 1):
          axs[i].set_title('Active', fontsize = 8, color = 'green' )


      elif (samples[batch_idx[i]][1][image_idx[i]].numpy() != pred[i]):
          if (samples[batch_idx[i]][1][image_idx[i]].numpy() == 0):
            axs[i].set_title('Non active', fontsize = 8, color = 'red' )
          elif (samples[batch_idx[i]][1][image_idx[i]].numpy() == 1):
            axs[i].set_title('Active', fontsize = 8, color = 'red' )



    plt.tight_layout()
    plt.show()

"""#Model 1: CNN Model"""

inp = Input(shape = (150, 150, 3))
c = ConvBlock(32, 2, activation = 'relu', batchnormalization = False, name = 'conv1')(inp)
c = ConvBlock(64, 2, activation = 'relu', batchnormalization = False , name = 'conv2')(c)
c = ConvBlock(128, 2, activation = 'relu', batchnormalization = False, name = 'conv3' )(c)

c = GlobalAveragePooling2D()(c)
c = Dense(64, activation = 'relu')(c)
c = Dense(128, activation = 'relu')(c)
out = Dense(1, activation = 'sigmoid')(c)

model = Model(inputs = inp, outputs = out)
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(), loss ='binary_crossentropy', metrics = [ 'AUC', 'Precision', 'Recall' , 'mae', 'mse'] )
my_callbacks = [PredictDuringTraining()]

#Plotting the model

plt.figure(dpi = 100)
visualkeras.layered_view(model, spacing = 50,  scale_z = 1, scale_xy = 2 , legend=True)

hist = model.fit(train_dataset,
                 epochs = 50 ,
                 batch_size = BATCH_SIZE,
                 validation_data= validation_dataset,
                 callbacks = my_callbacks )

"""#Model 1: Results"""

fig, axs = plt.subplots(2,3 , figsize = ( 21, 11 ), dpi = 200)

axs[0][0].grid(linestyle = 'dashdot')
axs[0][0].plot(hist.history['loss'] )
axs[0][0].plot(hist.history['val_loss'] )
axs[0][0].set_xlabel('epochs', fontsize = 8)
axs[0][0].legend(['train', 'val'], fontsize = 8)
axs[0][0].set_title('Loss', fontsize = 8)


axs[0][1].grid(linestyle = 'dashdot')
axs[0][1].plot(hist.history['mae'] )
axs[0][1].plot(hist.history['val_mae'] )
axs[0][1].set_xlabel('epochs', fontsize = 8)
axs[0][1].legend(['train', 'val'], fontsize = 8)
axs[0][1].set_title('MAE', fontsize = 8)

axs[0][2].grid(linestyle = 'dashdot')
axs[0][2].plot(hist.history['mse'] )
axs[0][2].plot(hist.history['val_mse'] )
axs[0][2].set_xlabel('epochs', fontsize = 8)
axs[0][2].legend(['train', 'val'], fontsize = 8)
axs[0][2].set_title('MSE', fontsize = 8)

axs[1][0].grid(linestyle = 'dashdot')
axs[1][0].plot(hist.history['auc'])
axs[1][0].plot(hist.history['val_auc'] )
axs[1][0].set_xlabel('epochs', fontsize = 8)
axs[1][0].legend(['train', 'val'], fontsize = 8)
axs[1][0].set_title('AUC', fontsize = 8)

axs[1][1].grid(linestyle = 'dashdot')
axs[1][1].plot(hist.history['precision'])
axs[1][1].plot(hist.history['val_precision'] )
axs[1][1].set_xlabel('epochs', fontsize = 8)
axs[1][1].legend(['train', 'val'], fontsize = 8)
axs[1][1].set_title('Precision', fontsize = 8)

axs[1][2].grid(linestyle = 'dashdot')
axs[1][2].plot(hist.history['recall'])
axs[1][2].plot(hist.history['val_recall'] )
axs[1][2].set_xlabel('epochs', fontsize = 8)
axs[1][2].legend(['train', 'val'], fontsize = 8)
axs[1][2].set_title('Recall', fontsize = 8)

"""#Model 1: Test Prediction, Evaluation"""

# Test set evaluation

test_eval = model.evaluate(test_dataset)

print('test auc : {0:.3f}'.format(test_eval[1]))
print('test precision : {0:.3f}'.format(test_eval[2]))
print('test recall : {0:.3f}'.format(test_eval[3]))

# Test set prediction

test_take1 =  test_dataset.take(-1)
test_take1_ = list(test_take1)
pred = model.predict(test_take1)
pred_ = np.round(pred)

y_test_take = []
for x in range(len(test_take1_)):
    y_test_take.extend(test_take1_[x][1].numpy())

# Plotting the confusion matrix of the CNN Model

cm = confusion_matrix(y_test_take, pred_)
cmd = ConfusionMatrixDisplay(cm, display_labels= ['non active','active'])
cmd.plot(cmap = 'BuPu',colorbar = False )

rp = classification_report(y_test_take, pred_)
print(rp)

"""#Model 2: CNN Feature Extraction and CatBoostClassifier"""

inp = Input(shape = (150, 150, 3))
c = ConvBlock(32, 2, activation = 'elu', batchnormalization = True, name = 'conv1')(inp)
c = ConvBlock(64, 2, activation = 'elu', batchnormalization = True , name = 'conv2')(c)
c = ConvBlock(128, 2, activation = 'elu', batchnormalization = True, name = 'conv3' )(c)
c = ConvBlock(150, 2, activation = 'elu', batchnormalization = True, name = 'conv4' )(c)
c = GlobalAveragePooling2D()(c)


model2 = Model(inputs = inp, outputs = c)
model2.summary()

model2.compile(optimizer = tf.keras.optimizers.Adam(), loss ='binary_crossentropy')
my_callbacks2 = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 0.0001, patience = 5 ) ]


#Plotting the model

plt.figure(dpi = 100)
visualkeras.layered_view(model2, spacing = 50,  scale_z = 1, scale_xy = 2 , legend=True)

hist2 = model2.fit(train_dataset_without_shuffle,
                   epochs = 100,
                   batch_size = BATCH_SIZE,
                    callbacks = my_callbacks2)

# Feature extraction

train_feat = model2.predict(train_dataset_without_shuffle)
val_feat = model2.predict(validation_dataset)
test_feat = model2.predict(test_dataset)

"""#Model 2: CatBoostClassifier for Classification

"""

cat_model = CatBoostClassifier(iterations = 2500)
cat_model.fit(train_feat, train_data['labels'], verbose = 500 )

cat_pred = cat_model.predict(test_feat)

# Plotting the confusion matrix of the CatBoostClassifier Model

cm2 = confusion_matrix(test_data['labels'], cat_pred)
cmd2 = ConfusionMatrixDisplay(cm2, display_labels= ['non active','active'])
cmd2.plot(cmap = 'BuPu',colorbar = False )

rp2 = classification_report(test_data['labels'], cat_pred)
print(rp2)

"""#Model 3: Ensemble Weighted Classifier"""

def weighted_classifier(pred1 , pred2, model1_weight, model2_weight):
  last_preds = []
  for i in range(len(pred1)):
    pred = pred1[i]*model1_weight + pred2[i]*model2_weight
    last_preds.append(pred[0])

  return np.round(last_preds)

# Plotting the confusion matrix of the Ensemble Model
# I have used 70% of the first model's prediction and 30% of the second model's prediction

weighted_pred = weighted_classifier(pred, cat_pred, 0.7, 0.3)

cm3 = confusion_matrix(test_data['labels'], weighted_pred)
cmd2 = ConfusionMatrixDisplay(cm3, display_labels= ['non active','active'])
cmd2.plot(colorbar = False )

rp3 = classification_report(test_data['labels'], weighted_pred)
print(rp3)
