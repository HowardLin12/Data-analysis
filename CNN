import os,shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
original_dataset_dir=r'F:\DL\HW2\DATA\Train_Image'
base_dir=r'F:\DL\HW2\DATA\Class'

if not os.path.isdir(base_dir):os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):os.mkdir(train_dir)

train_A_dir=os.path.join(train_dir,'A')
if not os.path.isdir(train_A_dir):os.mkdir(train_A_dir)

train_B_dir=os.path.join(train_dir,'B')
if not os.path.isdir(train_B_dir):os.mkdir(train_B_dir)

train_C_dir=os.path.join(train_dir,'C')
if not os.path.isdir(train_C_dir):os.mkdir(train_C_dir)

fnames=['A.{}.jpg'.format(i)for i in range(1944)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dst=os.path.join(train_A_dir,fname)
    shutil.copyfile(src, dst)


fnames=['B.{}.jpg'.format(i)for i in range(2255)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dst=os.path.join(train_B_dir,fname)
    shutil.copyfile(src, dst)


fnames=['C.{}.jpg'.format(i)for i in range(1928)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dst=os.path.join(train_C_dir,fname)
    shutil.copyfile(src, dst)

train_dir=os.path.join(base_dir,'train')
#Train model
import os,shutil
import os
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
df=pd.read_csv('train.csv')
base_dir=r'D:\DL\Class'
if not os.path.isdir(base_dir):os.mkdir(base_dir)
train_dir=os.path.join(base_dir,'train')
from keras_preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1/255)
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=20,target_size=(150,150),class_mode='categorical')

from keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import optimizers
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
conv_base.trainable=False

from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30)

model.save('mango')
#Test Model
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import os
model=load_model('mango_final.h5')
base_dir=r'D:\DL\HW2'
test_dir=os.path.join(base_dir,'test_class')
test_datagen=ImageDataGenerator(rescale=1/255)
test_generator=test_datagen.flow_from_directory(test_dir,batch_size=20,target_size=(150,150),class_mode='categorical')
accuracy, f1_score, precision, recall=model.evaluate_generator(test_generator,steps=50)
print(accuracy)
print(f1_score)
print(precision)
print(recall)
