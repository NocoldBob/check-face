
import tensorflow as tf  
import numpy as np  
import os  


train_dir = './data/train/'    
  
def get_files(file_dir):  
    ''''' 
    Args: 
        file_dir: file directory 
    Returns: 
        list of images and labels 
    '''  
    cats = []  
    label_cats = []  
    dogs = []  
    label_dogs = []  
    for file in os.listdir(file_dir):  
        name = file.split(sep='.')  
        if name[0]=='cat':  
            cats.append(file_dir + file)  
            label_cats.append(0)  
        else:  
            dogs.append(file_dir + file)  
            label_dogs.append(1)  
    # print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))  
      
    image_list = np.hstack((cats, dogs))  
    label_list = np.hstack((label_cats, label_dogs))  
      
    temp = np.array([image_list, label_list])  
    temp = temp.transpose()  
    np.random.shuffle(temp)  
      
    image_list = list(temp[:, 0])  
    label_list = list(temp[:, 1])  
    label_list = [int(i) for i in label_list]  
         
    return image_list, label_list  
  
  
 
def get_batch(image, label, image_W, image_H, batch_size, capacity):  
    ''''' 
    Args: 
        image: list type 
        label: list type
        指定图片参数 
        image_W: image width 
        image_H: image height 
        batch_size: batch size 
        capacity: the maximum elements in queue 
    Returns: 
        image_batch: 4D tensor [batch_size, width, height, 3（通道，rgb）], dtype=tf.float32 
        label_batch: 1D tensor [batch_size], dtype=tf.int32 
    '''  
      
    image = tf.cast(image, tf.string)  
    label = tf.cast(label, tf.int32)  
  
  
    # make an input queue  生成输入队列
    input_queue = tf.train.slice_input_producer([image, label])  
      
    label = input_queue[1]  
    image_contents = tf.read_file(input_queue[0])  
    image = tf.image.decode_jpeg(image_contents, channels=3)  
      
    ######################################  
    # data argumentation should go to here  
    ######################################  
      
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  #裁剪或者扩充图片
    image = tf.image.per_image_standardization(image)                       #标准化   
    image_batch, label_batch = tf.train.batch([image, label],  
                                                batch_size= batch_size,  
                                                num_threads= 64,     #县城数
                                                capacity = capacity)  #线程中的容纳元素
      
    #you can also use shuffle_batch   打乱图片顺序
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],  
#                                                      batch_size=BATCH_SIZE,  
#                                                      num_threads=64,  
#                                                      capacity=CAPACITY,  
#                                                      min_after_dequeue=CAPACITY-1)  
      
    label_batch = tf.reshape(label_batch, [batch_size])  
    # image_batch = tf.cast(image_batch, tf.float32)  
      
    return image_batch, label_batch  
 




#%%
import matplotlib.pyplot as plt 
  
BATCH_SIZE = 2              #每一个batch中的图片张数
CAPACITY = 256  
IMG_W = 208  
IMG_H = 208  
   
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'  
train_dir = './data/train/'   #存放数据路径

image_list, label_list = get_files(train_dir)  #得到路径
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)  
  
with tf.Session() as sess:  
    i = 0  #控制运行的步骤
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  
     #安全模式 
    try:  
        while not coord.should_stop() and i<1:  #CAPACITY执行步数
              
            img, label = sess.run([image_batch, label_batch])  
              
            # just test one batch  
            for j in np.arange(BATCH_SIZE):  
                print('label: %d' %label[j])    
                
                plt.imshow(img[j,:,:,:])  #4d数据，第一个控制一下
                # plt.show()
            i+=1  
              
    except tf.errors.OutOfRangeError:  
        print('done!')  
    finally:  
        coord.request_stop()  
    coord.join(threads)  
  
#%%
