import os
import numpy as np
import tensorflow as tf
import threading
from os import listdir
from os.path import isfile, join, isdir
import time
import random

from PIL import Image
import scipy.io as sio
import scipy.misc as sm
#import skimage
#import skimage.transform
#from skimage import io

FLAGS = tf.app.flags.FLAGS

# ===========================
#   Data Parameters
# ===========================
tf.app.flags.DEFINE_string("dataset", "sandbox","pick the dataset in data_root from which your movies can be found.")
tf.app.flags.DEFINE_integer("batch_size", 16, "The size of the minibatch used for training.")
tf.app.flags.DEFINE_string("data_root", "/home/klaas/pilot_data", "Define the root folder of the different datasets.")
tf.app.flags.DEFINE_integer("num_threads", 1, "The number of threads for loading one minibatch.")

datasetdir = join(FLAGS.data_root, FLAGS.dataset)
full_set = {}
im_size=(250,250,3)

def load_set(data_type):
  """Load a type (train, val or test) of set in the set_list
  as a tuple: first tuple element the directory of the fligth 
  and the second the number of images taken in that flight
  """
  set_list = []
  f = open(join(datasetdir, data_type+'_set.txt'), 'r')
  lst = [ l[:-1] for l in f.readlines() ]
  for run_dir in lst:
    num_img = len(listdir(join(run_dir,'RGB')))
    num_cont = len(open(join(run_dir,'control_info.txt'), 'r').readlines())
    num = min(num_img, num_cont)-1
    set_list.append((run_dir, num))
    if not isfile(join(run_dir,'RGB','{0:010d}.jpg'.format(num))):
      print('ERROR:',run_dir,' imgnum: ',num_img, ' cont: ',num_cont, 'num: ', num)
  f.close()
  return set_list

def prepare_data(size):
  global im_size, full_set
  '''Load lists of tuples refering to images from which random batches can be drawn'''
  train_set = load_set('train')
  val_set=load_set('val')
  test_set=load_set('test')
  full_set={'train':train_set, 'val':val_set, 'test':test_set}
  im_size=size
  
def generate_batch(data_type):
  """ Generator object that gets a random batch when next() is called
  """
  data_set=full_set[data_type]
  number_of_frames = sum([t[1] for t in data_set])
  # When there is that much data applied that you can get more than 100 minibatches out
  # stick to 100, otherwise one epoch takes too long and the training is not update
  # regularly enough.
  max_num_of_batch = {'train':100, 'val':10, 'test':1000}
  number_of_batches = min(int(number_of_frames/FLAGS.batch_size),max_num_of_batch[data_type])
  b=0
  while b < number_of_batches:
    if b>0 and b%10==0:
      print('batch ',b,' of ',number_of_batches)
    #print('batch ',cnt,' of ',number_of_batches)
    ok = True
    # Single threaded implementation
    if False:
      im_b = []
      trgt_b = []
      for j in range(FLAGS.batch_size):
        # choose random run:
        run_ind = random.choice(range(len(data_set)))
        # choose random image:
        im_ind = random.choice(range(data_set[run_ind][1]))
        # load image
        img_file = join(data_set[run_ind][0],'RGB', '{0:010d}.jpg'.format(im_ind))
        im = Image.open(img_file)
        im = sm.imresize(im,im_size,'nearest')
        im = im * 1/255.
        im_b.append(im)
        # load target
        control_file = open(join(data_set[run_ind][0],'control_info.txt'),'r')
        control_list = [ l[:-1] for l in control_file.readlines()]
        # get the target yaw turn from the control and map between -1 and 1
        control = max(min(1, control_list[im_ind].split(' ')[6]), -1)
        trgt_b.append([control])
        
    else: #Multithreaded implementation
      im_b = np.zeros((FLAGS.batch_size,im_size[0],im_size[1],im_size[2])) #[] #list(range(FLAGS.batch_size)) #
      trgt_b = np.zeros((FLAGS.batch_size, 1)) #[] #list(range(FLAGS.batch_size)) #
      batch_indices = []
      checklist = []
      thread_indices = list(range(FLAGS.batch_size))
      for j in range(FLAGS.batch_size):
        # choose random run:
        run_ind = random.choice(range(len(data_set)))
        # choose random image:
        im_ind = random.choice(range(data_set[run_ind][1]))
        batch_indices.append((run_ind, im_ind))
      def load_image_and_target(coord, batch_indices, thread_indices, im_b, trgt_b, checklist):
        while not coord.should_stop():
          try:
            loc_ind = thread_indices.pop()
            #print('------------',loc_ind)
            #loc_ind, run_ind, img_ind = batch_indices.pop()
            run_ind, img_ind = batch_indices[loc_ind]
            #print('------------',loc_ind)
            # load image
            img_file = join(data_set[run_ind][0],'RGB', '{0:010d}.jpg'.format(im_ind))
            im = Image.open(img_file)
            im = sm.imresize(im,im_size,'nearest')
            im = im * 1/255.
            im_b[loc_ind,:,:,:]=im
            #im_b[loc_ind]=im
            #im_b.append(im)
            # load target
            control_file = open(join(data_set[run_ind][0],'control_info.txt'),'r')
            control_list = [ l[:-1] for l in control_file.readlines()]
            # get the target yaw turn from the control and map between -1 and 1
            #control = max(min(1, control_list[im_ind].split(' ')[6]), -1)
            control = float(control_list[im_ind].split(' ')[6])
            if abs(control) > 1: control = np.sign(control)
            trgt_b[loc_ind, :]=[control]
            #trgt_b.append([control])
            checklist.append(True)
          except IndexError as e:
            #print('batch_loaded, wait to stop', e)
            coord.request_stop()
          except Exception as e:
            print('Problem in loading data: ',e)
            checklist.append(False)
            coord.request_stop()
      try:
        coord=tf.train.Coordinator()
        #print(FLAGS.num_threads)
        threads = [threading.Thread(target=load_image_and_target, args=(coord, batch_indices, thread_indices, im_b, trgt_b, checklist)) for i in range(FLAGS.num_threads)]
        for t in threads: t.start()
        coord.join(threads, stop_grace_period_secs=5)
      except RuntimeError as e:
        print("threads are not stopping...",e)
      else:
        if len(checklist) != sum(checklist): ok=False
    if ok: b+=1
    yield ok, np.asarray(im_b), np.asarray(trgt_b)
    
#### FOR TESTING ONLY: DELETE LATER
if __name__ == '__main__':
  def print_dur(start_time):
    duration = (time.time()-start_time)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return "time: %dh:%02dm:%0.5fs" % (h, m, s)
  prepare_data((299,299,3))
  start_time=time.time()
  for ok, imb, trgtb in generate_batch('train'):
    if not ok: print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!problem...')#print(imb.shape, trgtb.shape)
    #print imb.shape, trgtb.shape
    print trgtb
    pass
  print('loading time one episode: ', print_dur(start_time))
  