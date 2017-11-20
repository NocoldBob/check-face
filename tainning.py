import os  
import numpy as np  
import tensorflow as tf  
import input_data     
import model  
import pygame  
  
#%%  
  
  
N_CLASSES = 2  
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.  
IMG_H = 208  
BATCH_SIZE = 32  
CAPACITY = 2000  
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k  
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001  
  
  
  
  
#%%  
def run_training():  
      
  
    train_dir = './data/train/'    
    
    logs_train_dir = './logs/train/'  
    train, train_label = input_data.get_files(train_dir)  
      
    train_batch, train_label_batch = input_data.get_batch(train,  
                                                          train_label,  
                                                          IMG_W,  
                                                          IMG_H,  
                                                          BATCH_SIZE,   
                                                          CAPACITY)        
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)  
    train_loss = model.losses(train_logits, train_label_batch)          
    train_op = model.trainning(train_loss, learning_rate)  
    train__acc = model.evaluation(train_logits, train_label_batch)  
         
    summary_op = tf.summary.merge_all()  
    sess = tf.Session()  
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)  
    saver = tf.train.Saver()  
      
    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
      
    try:  
        for step in np.arange(MAX_STEP):  
            if coord.should_stop():  
                    break  
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])  
                 
            if step % 10 == 0:  
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))  
                summary_str = sess.run(summary_op)  
                train_writer.add_summary(summary_str, step)  
              
            if step % 500 == 0 or (step + 1) == MAX_STEP:  
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
                saver.save(sess, checkpoint_path, global_step=step)  
            if step>300 and tra_loss == 0.00:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
                saver.save(sess, checkpoint_path, global_step=step) 
                coord.request_stop()  
                  
    except tf.errors.OutOfRangeError:  
        print('Done training -- epoch limit reached')
        file=r'./failed.mp3'
        pygame.mixer.init()
        track = pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        time.sleep(10)
        pygame.mixer.music.stop()  
    finally:  
        coord.request_stop()  
        file=r'./successful.mp3'
        pygame.mixer.init()
        track = pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        time.sleep(10)
        pygame.mixer.music.stop()  
    coord.join(threads)  
    sess.close()  
run_training()     
 
  
