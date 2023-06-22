# %% [markdown]
# In this notebook, we'll go through the steps to train a CRNN (CNN+RNN) model for handwriting recognition. The model will be trained using the CTC(Connectionist Temporal Classification) loss.
# 
# 
# ![Webp.net-resizeimage.jpg](attachment:Webp.net-resizeimage.jpg)

# %% [markdown]
# Lets start by importing the necessary libraries. I'll be using Keras for building our CRNN model.

# %%
import os
import cv2
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
tf.config.run_functions_eagerly(True)

# %% [markdown]
# ## Load and view data

# %%
train = pd.read_csv('/kaggle/input/handwriting-recognition/written_name_train_v2.csv')
valid = pd.read_csv('/kaggle/input/handwriting-recognition/written_name_validation_v2.csv')

# %%
train.shape

# %%
plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = '/kaggle/input/handwriting-recognition/train_v2/train/'+train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    plt.title(train.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# %% [markdown]
# ## Cleaning Data

# %% [markdown]
# Let's check for NaNs in our label.

# %%
print("Number of NaNs in train set      : ", train['IDENTITY'].isnull().sum())
print("Number of NaNs in validation set : ", valid['IDENTITY'].isnull().sum())

# %%
train.dropna(axis=0, inplace=True)
valid.dropna(axis=0, inplace=True)

# %% [markdown]
# Also, there are some images in our data with the label 'UNREADABLE'. Lets check those images and remove them.

# %%
unreadable = train[train['IDENTITY'] == 'UNREADABLE']
unreadable.reset_index(inplace = True, drop=True)

plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = '/kaggle/input/handwriting-recognition/train_v2/train/'+unreadable.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap = 'gray')
    plt.title(unreadable.loc[i, 'IDENTITY'], fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# %%
train = train[train['IDENTITY'] != 'UNREADABLE']
valid = valid[valid['IDENTITY'] != 'UNREADABLE']

# %% [markdown]
# There are some labels which are in lowercase. To maintain uniformity in the labels, I convert all the labels to uppercase.

# %%
train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()

# %% [markdown]
# Reset the index and we are done with cleaning. 

# %%
train.reset_index(inplace = True, drop=True) 
valid.reset_index(inplace = True, drop=True)

# %%
train.shape[0]/128

# %% [markdown]
# ## Preprocessing and preparing the images for training

# %% [markdown]
# * The images are loaded as grayscale and reshaped to width 256 and height 64.  
# * The width and height are cropped if they are greater than 256 and 64 respectively. If they are smaller, then the image is padded with white pixels. Finally the image is rotated clockwise to bring the image shape to (x, y). 
# * The image is then normalized to range [0, 1]

# %%
def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# %% [markdown]
# The model will be trained on 30000 images and validate on 3000 images

# %%
# train_size = 30000
valid_size= 3000

# %%
valid_x = []

for i in range(valid_size):
    img_dir = '/kaggle/input/handwriting-recognition/validation_v2/validation/'+valid.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    valid_x.append(image)

# %%
# train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

# %% [markdown]
# ## Preparing the labels for CTC Loss
# 
# Learn more about CTC loss and why its amazing for text recognition from [here](https://theailearner.com/2019/05/29/connectionist-temporal-classificationctc/).
# 
# The labels have to be converted to numbers which represent each character in the training set. The 'alphabets' consist of A-Z and three special characters (-  '  and space). 

# %%
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels
batch_size = 128

def label_to_num(label):
    label_num = []
    for ch in label:
        
            label_num.append(alphabets.find(ch) if alphabets.find(ch)!=-1 else alphabets.find('-'))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

# %%
name = 'JEBASTIN'
print(name, '\n',label_to_num(name))

# %% [markdown]
# * **train_y** contains the true labels converted to numbers and padded with -1. The length of each label is equal to max_str_len. 
# * **train_label_len** contains the length of each true label (without padding) 
# * **train_input_len** contains the length of each predicted label. The length of all the predicted labels is constant i.e number of timestamps - 2.  
# * **train_output** is a dummy output for ctc loss. 
# 

# %% [markdown]
# # Use gnerator for same purpose

# %%
train.shape

# %%
# path = '/kaggle/input/handwriting-recognition/train_v2/train/'

# counts = []
# for row, val in train.iloc[37000:40000].iterrows():
# #     pass
#     img_dir = os.path.join(path, val['FILENAME'])
#     image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
#     counts.append(len(val['IDENTITY']))
# print(counts)

# %%
len(train.iloc[39128]['IDENTITY'])

# %%
directory = '/kaggle/input/handwriting-recognition/train_v2/train/'

def image_generator(path):
        for row, val in  train.iterrows(): #need improvement
#             try:
                img_dir = os.path.join(path, val['FILENAME'])
                image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                image = preprocess(image)[..., np.newaxis]
        #         print(image.shape)
                image = image/255.
                if image.shape !=(256,64,1):
                    break
        #         train_label_len[i] = len(train.loc[i, 'IDENTITY'])
                if (len(val['IDENTITY'])>max_str_len):
                    continue
                labels = label_to_num(val['IDENTITY'])    
                yield (image, labels)
#             except Exception as e:
#                 continue


# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE

# %%


# %%
# # dataset = tf.data.Dataset.from_generator(lambda: image_generator(directory), output_types=())
# directory = '/kaggle/input/handwriting-recognition/train_v2/train/'

dataset = tf.data.Dataset.from_generator(lambda: image_generator(directory), (tf.float32, tf.float32))# , ([256, 64,1], [None]))
def foo(*val):
    xl = val[0]
    lab = val[1]
    lab_len = len(lab)
    one_pad = -tf.ones((max_str_len - lab_len))#*(-1)
    y = tf.concat([lab, one_pad], -1)
    
    return xl, y, [num_of_timestamps-2], [lab_len], [0]
                  
dataset = dataset.map(foo).batch(batch_size)
dataset_final = dataset.map(lambda *x: (x[0:-1],x[-1])).prefetch(AUTOTUNE)


# %%
# # c = 0
# for j in dataset_final:
# #     c+=1
# # #     if c>1:
#     print(len(j))
#     print(j[0][1])
# #     print(j[0][0].shape, j[0][1].shape,  j[0][2].shape, j[0][3].shape, j[1].shape)
# #     print(j[0][0].shape, j[0][1].shape,  j[0][2], j[0][3], j[1].shape)
#     break

# %%
valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(valid.loc[i, 'IDENTITY'])
    valid_y[i, 0:len(valid.loc[i, 'IDENTITY'])]= label_to_num(valid.loc[i, 'IDENTITY'])    

# %%
# print('True label : ',train.loc[100, 'IDENTITY'] , '\ntrain_y : ',train_y[100],'\ntrain_label_len : ',train_label_len[100], 
#       '\ntrain_input_len : ', train_input_len[100])

# %% [markdown]
# ## Building our model
# 

# %%
input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)

# CNN to RNN
inner = Reshape(target_shape=((num_of_timestamps, 1024)), name='reshape')(inner)
inner = Dense(num_of_timestamps, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

## RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()

# %% [markdown]
# The output shape of the predictions is (64, 30). The model predicts words of 64 characters and each character contains the probability of the 30 alphabets which we defined earlier.  

# %%
# the ctc loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# %%
labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

# %% [markdown]
# ## Train our model

# %%
# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = 0.0001))

# %%

model_final.fit(dataset_final, epochs=8,verbose=1)
# model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output, 
#                 validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
#                 epochs=60, batch_size=128)

# %%
model.save('/kaggle/working/model', save_format='h5')


# %%
# model_final.load('/kaggle/working/model', save_format='h5')
from tensorflow import keras

# final_model = keras.models.load_model('/kaggle/working/model', compile=False)


# %%
my_model = keras.models.load_model('/kaggle/working/model')#, compile=False)

# %%
# from IPython.display import FileLink
# import os 
# os.chdir(r'/kaggle/working')
# FileLink(r'model')

# %%
# train_generator = get_batch_generator(image_generator(directory),batch_examples)

# model_final.fit(dataset_final.repeat(2), epochs=9)

# %%
# batch_examples = 128
# train_generator = get_batch_generator(image_generator(directory),batch_examples)
# .fit_generator(
#     generator=train_generator,
#     steps_per_epoch=train.shape[0]//batch_examples,
#     epochs=10,)

# %% [markdown]
# ## Check model performance on validation set

# %%
preds = my_model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))

# %%
y_true = valid.loc[0:valid_size, 'IDENTITY']
correct_char = 0
total_char = 0
correct = 0

for i in range(valid_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))

# %% [markdown]
# ## Some predictions on test set

# %%
test = pd.read_csv('/kaggle/input/handwriting-recognition/written_name_test_v2.csv')

plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = '/kaggle/input/handwriting-recognition/test_v2/test/'+test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    image = preprocess(image)
    image = image/255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)

# %% [markdown]
# ## transfer learning

# %%
import string
import json


# %%
batch_size = 8
alphabets = string.digits +"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

# %%
width = 256
height = 64
# num_of_timestamps = 50
# max_str_len = num_of_timestamps
rnn_steps_to_discard = 2
all_data = '/kaggle/input/tags-licenseplate/ocr-licence-plate.csv'
data_all = pd.read_csv(all_data)

# %%
def yolo_to_abs(data_dict):#, scale=100.0):
    cord = {}

    original_width = data_dict['original_width']
    original_height = data_dict['original_height']
    
    pixel_x = int(data_dict['x']/100. * original_width)
    pixel_y = int(data_dict['y']/100. * original_height)
    pixel_width = int(data_dict['width']/100. * original_width)
    pixel_height = int(data_dict['height']/100. * original_height)
#     return[pixel_x, pixel_y, pixel_width, pixel_height]
    cord['x1'] = pixel_x #- pixel_width #/2)*original_width
    cord['y1'] = pixel_y #- pixel_height#/2)*original_height
    cord['x2'] = pixel_x + pixel_width#/2)*original_width
    cord['y2'] = pixel_y + pixel_height#/2)*original_height
    return cord

# %%
def preprocess_2(image, width: int, height: int, cval: int = 255, mode="letterbox", return_scale=False,):
    """Obtain a new image, fit to the specified size.
    Args:
        image: The input image
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image
        return_scale: Whether to return the scale used for the image
    Returns:
        The new image
    """
    fitted = None
    x_scale = width / image.shape[1]
    y_scale = height / image.shape[0]
    if x_scale == 1 and y_scale == 1:
        fitted = image
        scale = 1
    elif (x_scale <= y_scale and mode == "letterbox") or (
        x_scale >= y_scale and mode == "crop"
    ):
        scale = width / image.shape[1]
        resize_width = width
        resize_height = (width / image.shape[1]) * image.shape[0]
    else:
        scale = height / image.shape[0]
        resize_height = height
        resize_width = scale * image.shape[1]
    if fitted is None:
        resize_width, resize_height = map(int, [resize_width, resize_height])
        if mode == "letterbox":
            fitted = np.zeros((height, width, 3), dtype="uint8") + cval
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted[: image.shape[0], : image.shape[1]] = image[:height, :width]
        elif mode == "crop":
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted = image[:height, :width]
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")
    if not return_scale:
        return fitted
    return fitted, scale

# %%
def image_generator_2(path, pd_data):
    for row, val in  pd_data.iterrows():                         # need improvement
        img_dir = os.path.join(path, val['ocr'].split('-')[-1])
        image = cv2.imread(img_dir, cv2.IMREAD_ANYCOLOR)
#             image = preprocess(image)[..., np.newaxis]
        try:
            labels = json.loads(val['transcription'])
        except ValueError:
            labels = str(val['transcription'])
        for num, bbox in enumerate(json.loads(val['bbox'])):
            cart_cord = yolo_to_abs(bbox)
            crop_img = image[cart_cord['y1']:cart_cord['y2'],
                             cart_cord['x1']:cart_cord['x2']]
            process_img = preprocess_2(crop_img, width=width,height=height)
            process_img = cv2.rotate(process_img, cv2.ROTATE_90_CLOCKWISE)
            process_img = cv2.cvtColor(process_img, cv2.COLOR_RGB2GRAY).astype("float32")[..., np.newaxis]
            process_img /=255
            if isinstance(labels, list):
                yield process_img, label_to_num(labels[num])
            else:
                yield process_img, label_to_num(str(labels))
#             image = image/255.
#             if image.shape !=(256,64,1):
#                 break
    #         train_label_len[i] = len(train.loc[i, 'IDENTITY'])
#             if (len(val['IDENTITY'])>max_str_len):
#                 continue
#             labels = val['transcription'] 
#             labels = label_to_num(val['IDENTITY'])    
#             yield (image, labels)
#         except Exception as e:
#             continue
def data_joiner(*val):
    xl = val[0]
    lab = val[1]
    lab_len = len(lab)
    one_pad = -tf.ones((max_str_len - lab_len - rnn_steps_to_discard))#*(-1)
    y = tf.concat([lab, one_pad], -1)
    
    return xl, y, [num_of_timestamps-2], [lab_len], [0]


# %%
int(len(data_all)*.8)

# %%
cuttof = 220
train = data_all.iloc[:cuttof]
test = data_all.iloc[cuttof:]

# %%
batch_size = 8

directory = '/kaggle/input/car-plate-detection/images/'
                  
dataset = tf.data.Dataset.from_generator(lambda: image_generator_2(directory, train), (tf.float32, tf.float32))# , ([256, 64,1], [None]))
dataset = dataset.map(data_joiner).batch(batch_size)
dataset_train= dataset.map(lambda *x: (x[0:-1],x[-1]))

# %%
# for i in dataset_train:
#     print(i)
#     break

# %%
# plt.plot(i[0][0][0,:,:,:])
# plt.imshow(i[0][0][0])
plt.imshow(i[0][0][0,:,:,0], cmap='gray')

# %%

decoder_layer = Dense(len(alphabets) + 1, kernel_initializer='he_normal',
                           activation='softmax',  name='fc_12')(my_model.layers[-3].output)
x = Lambda(lambda x: x[:, rnn_steps_to_discard:])(decoder_layer)


new_model = Model(inputs=my_model.inputs, outputs=x)
# prediction_model = keras.models.Model(inputs=new_model.inputs, outputs=CTCDecoder()(new_model.output))

# %%
# for layer in new_model.layers[:-7]:
# #     print(layer)
#      layer.trainable = False

# %%
labels = Input(name="labels", shape=[new_model.output_shape[1]], dtype="float32")

label_length = Input(shape=[1])
input_length = Input(shape=[1])

loss = Lambda(lambda x: K.ctc_batch_cost(y_true=x[0], y_pred=x[1], input_length=x[2], label_length=x[3])
                          )([labels, new_model.output, input_length, label_length])
training_model =  Model(inputs=[new_model.input, labels, input_length, label_length], outputs=loss)

# %%
# for layer in new_model.layers[:-9]:
# #     print(layer)
#      layer.trainable = False

# %%
training_model.summary()

# %%
# kwargs = {}
# kwargs["optimizer"] = Adam(lr = 0.0001) #"RMSprop" #Adam
# kwargs["loss"] = lambda _, y_pred: y_pred
training_model.compile(loss=lambda _, y_pred: y_pred,optimizer=Adam(lr = 0.0001))

# %%
training_model.fit(dataset_train, epochs=8)#,verbose=1)

# %%
plt.imshow(i[0][0][0,:,:,0], cmap='gray')

# %%
batch_size = 1

dataset = tf.data.Dataset.from_generator(
    lambda: image_generator_2(directory, test), (tf.float32, tf.float32)).map(data_joiner).batch(batch_size)

dataset_test= dataset.map(lambda *x: (x[0:-1],x[-1]))

# %%
for smaple_test in dataset_test:
#     print(smaple_test)
    break

# %%
# plt.imshow(smaple_test[0][0][0])
# plt.plot(i[0][0][0,:,:,:])
# plt.imshow(i[0][0][0])
plt.imshow(smaple_test[0][0][0,:,:,0], cmap='gray')

# %%
labels = []
test_images = []
for X,y in dataset_test:
    test_images.append(X[0].numpy()[0])
    labels.append(X[1])

# %%
preds = new_model.predict(dataset_test)

preds = new_model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))

# %%
decoded = K.get_value(K.ctc_decode(v, input_length=np.ones(v.shape[0])*v.shape[1], 
                                   greedy=True)[0][0])

# %%
dataset_test.take(1)

# %%
preds = new_model.predict(test_images)

# %%
len(labels)

# %%
preds.shape

# %%
preds = new_model.predict(test_images)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], 
                                   greedy=True)[0][0])

# %%
prediction = []
for i in range(len()):
    prediction.append(num_to_label(decoded[i]))

# %% [markdown]
# # bentch mark

# %%
y_true = test.loc[0:valid_size, 'IDENTITY']
correct_char = 0
total_char = 0
correct = 0

for x ,y in range(valid_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))

# %%
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='/kaggle/input/keras-ocr/tflite/float16/2/2.tflite')


# %%
dir(interpreter)

# %%
interpreter.get_input_details()

# %%
interpreter.allocate_tensors()  # Needed before execution!

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.



# %%
path = '/kaggle/input/car-plate-detection/images'
test_example = test.iloc[0]
# val['ocr'].split('-')[-1])
# path = test_exaple
img_dir = os.path.join(path, test_example['ocr'].split('-')[-1])
image = cv2.imread(img_dir, cv2.IMREAD_ANYCOLOR)
labels = test_example['transcription']

for num, bbox in enumerate(json.loads(test_example['bbox'])):
    cart_cord = yolo_to_abs(bbox)
    crop_img = image[cart_cord['y1']:cart_cord['y2'],
                     cart_cord['x1']:cart_cord['x2']]
    process_img = preprocess_2(crop_img, width=200, height=31)
#     process_img = cv2.rotate(process_img, cv2.ROTATE_90_CLOCKWISE)
    process_img = cv2.cvtColor(process_img, cv2.COLOR_RGB2GRAY).astype("float32")[..., np.newaxis]
    process_img /=255


# %%
arr = process_img[np.newaxis, ...]

# %%
interpreter.set_tensor(input['index'], arr)
interpreter.invoke()
result = interpreter.get_tensor(output['index'])

# %%
DEFAULT_ALPHABET = string.digits + string.ascii_lowercase
def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=DEFAULT_ALPHABET[ch]
    return ret

# %%
result[0]

# %%
num_to_label(result[0])

# %%
labels

# %%



