import PIL
from keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from IPython.display import Image
import matplotlib.pyplot as plt
import os
from tensorflow.keras import callbacks
import pandas as pd
#fastest but dirty way to speed hack
from joblib import Parallel, delayed

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 16
EPOCHS = 200

#Train
dataframe = pd.read_csv('/home/yupaporn/codes/USAI/Traindf_fold1_3.csv')
base_dir = '/media/tohn/SSD/Images/Image1'
os.chdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
#validation
valframe = pd.read_csv( '/home/yupaporn/codes/USAI/Validationdf_fold1_3.csv')
validation_dir = os.path.join(base_dir, 'validation')

dataframe_train1HOT = pd.get_dummies(dataframe, columns=['Sub_class'], prefix=['Sub_class'])
dataframe_test1HOT = pd.get_dummies(valframe, columns=['Sub_class'], prefix=['Sub_class'])

#load model
import efficientnet.tfkeras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model_dir = '/media/tohn/SSD/ModelTrainByImages/R1_1/models/B5_R1_15AB_relu_1FC_3.h5' 

model = load_model(model_dir)
height = width = model.input_shape[1]

x = model.get_layer('top_activation').output
# prediction_layer = model.output
#predict angle branch
global_average_layer2 = layers.GlobalAveragePooling2D()(x)
dropout_layer_2 = layers.Dropout(0.50)(global_average_layer2)
prediction_layer2 = layers.Dense(14, activation='sigmoid',name='Pred_View')(dropout_layer_2)

model2 = models.Model(inputs= model.input, outputs=[prediction_layer2]) 


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe = dataframe_train1HOT,
        directory = train_dir,
        x_col = 'Path Crop',
        y_col = ['Sub_class_AB01','Sub_class_AB02','Sub_class_AB03','Sub_class_AB04','Sub_class_AB05',
                 'Sub_class_AB06','Sub_class_AB07','Sub_class_AB081','Sub_class_AB082','Sub_class_AB083','Sub_class_AB09','Sub_class_AB10',
                 'Sub_class_AB11','Sub_class_AB12'],
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='multi_output')

test_generator = test_datagen.flow_from_dataframe(
        dataframe = dataframe_test1HOT,
        directory = validation_dir,
        x_col = 'Path Crop',
        y_col = ['Sub_class_AB01','Sub_class_AB02','Sub_class_AB03','Sub_class_AB04','Sub_class_AB05',
                 'Sub_class_AB06','Sub_class_AB07','Sub_class_AB081','Sub_class_AB082','Sub_class_AB083','Sub_class_AB09','Sub_class_AB10',
                 'Sub_class_AB11','Sub_class_AB12'],
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='multi_output')

# os.chdir('/media/tohn/SSD/ModelTrainByImages/R2_Full')

# root_logdir = '/media/tohn/SSD/ModelTrainByImages/R2_Full/mylogsB5_15AB_Full_MultiLabel'
# def get_run_logdir():
#     import time
#     run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
#     return os.path.join(root_logdir,run_id)
# run_logdir = get_run_logdir()

# tensorboard_cb = callbacks.TensorBoard(log_dir = run_logdir)


# # os.makedirs("./models_6", exist_ok=True)

# def avoid_error(gen):
#     while True:
#         try:
#             data, labels = next(gen)
#             yield data, labels
#         except:
#             pass

#Unfreez
model2.trainable = True
set_trainable = False
for layer in model2.layers:
    if layer.name == 'block5a_se_excite':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model2.trainable_weights))  

#Training
# Define our metrics
train_loss = metrics.Mean('train_loss', dtype=tf.float32)
# train_loss1 = metrics.Mean('train_loss1', dtype=tf.float32)
# train_loss2 = metrics.Mean('train_loss2', dtype=tf.float32)
train_accuracy = metrics.CategoricalAccuracy('train_accuracy_Class')
# train_accuracy2 = metrics.CategoricalAccuracy('train_accuracy_Sub')


test_loss = metrics.Mean('test_loss', dtype=tf.float32)
# test_loss1 = metrics.Mean('test_loss1', dtype=tf.float32)
# test_loss2 = metrics.Mean('test_loss2', dtype=tf.float32)
test_accuracy = metrics.CategoricalAccuracy('test_accuracy_Class')
# test_accuracy2 = metrics.CategoricalAccuracy('test_accuracy_Sub')


optimizer=optimizers.RMSprop(lr=2e-5)
loss_object = losses.CategoricalCrossentropy()
# AbNorweight = 0.75


@tf.function
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        #loss2 = loss_object2(y_train[1], predictions[1])
        #loss = (AbNorweight*loss1) + ((1-AbNorweight)*loss2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
#     train_loss1(loss1)
    #train_loss2(loss2)
    train_accuracy(y_train, predictions)
    #train_accuracy2(y_train[1], predictions[1])

@tf.function    
def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    #loss2 = loss_object2(y_test[1], predictions[1])
#     loss = (AbNorweight*loss1) + ((1-AbNorweight)*loss2)

    test_loss(loss)
#      test_loss1(loss1)
    #test_loss2(loss2)
    test_accuracy(y_test, predictions)
    #test_accuracy2(y_test[1], predictions[1])
    
def load_training_data():

    data = next(train_generator)
    X = data[0]
    Y1 = np.column_stack((data[1][0], data[1][1], data[1][2], data[1][3],data[1][4], data[1][5], 
                          data[1][6], data[1][7],data[1][8], data[1][9], data[1][10], data[1][11], data[1][12], data[1][13]))
    #Y2 = np.column_stack((data[1][15], data[1][16], data[1][17], data[1][18], data[1][19]))
    
    return (X, Y1)

def load_test_data():

    data = next(test_generator)
    X = data[0]
    Y1 = np.column_stack((data[1][0], data[1][1], data[1][2], data[1][3],data[1][4], data[1][5], 
                          data[1][6], data[1][7],data[1][8], data[1][9], data[1][10], data[1][11], data[1][12], data[1][13]))
    #Y2 = np.column_stack((data[1][15], data[1][16], data[1][17], data[1][18], data[1][19]))
    
    return (X, Y1)

 #tensorboard
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/media/tohn/SSD/ModelTrainByImages/R2_1/mylogsnow_15AB_1_3/' + current_time + '/train'
test_log_dir = '/media/tohn/SSD/ModelTrainByImages/R2_1/mylogsnow_15AB_1_3/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

numittrain = (len(dataframe_train1HOT)//batch_size)+1
numittest = (len(dataframe_test1HOT)//batch_size)+1

prefetch_test = Parallel(n_jobs=10,backend='multiprocessing')(delayed(load_test_data)() for i in range(numittest))
                        
                         
for epoch in range(EPOCHS):
    prefetch_data =  Parallel(n_jobs=10,backend='multiprocessing')(delayed(load_training_data)() for i in range(numittrain))
    for step in range(numittrain):
        (X,Y) = prefetch_data[step]
        train_step(model2, optimizer, X, Y)
        for step_v in range(numittest):
            (X,Y) = prefetch_test[step_v]
            test_step(model2, X, Y)
        if step % 10 == 0:
            print(f'Epoch {epoch+1}, Step:{step+1}/{numittrain+1}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}, Val Loss: {test_loss.result():.4f}, Val Acc: {test_accuracy.result():.4f}')

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        #tf.summary.scalar('AB_loss', train_loss.result(), step=epoch)
        #tf.summary.scalar('Sub_loss', train_loss2.result(), step=epoch)
        tf.summary.scalar('AB_acc', train_accuracy.result(), step=epoch)
        #tf.summary.scalar('Sub_acc', train_accuracy2.result(), step=epoch)

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        #tf.summary.scalar('AB_loss', test_loss.result(), step=epoch)
        #tf.summary.scalar('Sub_loss', test_loss2.result(), step=epoch)
        tf.summary.scalar('AB_acc', test_accuracy.result(), step=epoch)
        #tf.summary.scalar('Sub_acc', test_accuracy2.result(), step=epoch)                      
                         
                         
# model2.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.RMSprop(lr=2e-5),
#               metrics=['acc'])

# run_logdir = get_run_logdir()

# tensorboard_cb = callbacks.TensorBoard(run_logdir)
# #early_stop_cb = callbacks.EarlyStopping(monitor='val_acc', patience=66, mode= 'max')

# history = model2.fit_generator(
#       avoid_error(train_generator),
#       steps_per_epoch= len(dataframe)//batch_size,
#       epochs=epochs,
#       validation_data=avoid_error(test_generator), 
#       validation_steps= len(valframe) //batch_size,
#       callbacks = [tensorboard_cb])

model2.save('/media/tohn/SSD/ModelTrainByImages/R2_Full/models/B5R2_15AB_Full_MultiLabel_fold1_3')
      