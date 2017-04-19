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
tf.app.flags.DEFINE_string("dataset", "mix","pick the dataset in data_root from which your movies can be found.")
tf.app.flags.DEFINE_integer("batch_size", 16, "The size of the minibatch used for training.")
tf.app.flags.DEFINE_string("data_root", "/esat/qayd/kkelchte/pilot_data", "Define the root folder of the different datasets.")
tf.app.flags.DEFINE_integer("num_threads", 4, "The number of threads for loading one minibatch.")
tf.app.flags.DEFINE_float("mean", 0, "Define the mean of the input data for centering around zero. Esat data:0.2623")
tf.app.flags.DEFINE_float("std", 1, "Define the standard deviation of the data for normalization. Esat data:0.1565")

datasetdir = join(FLAGS.data_root, FLAGS.dataset)
full_set = {}
im_size=(250,250,3)
de_size = (55,74)
def load_set(data_type):
  """Load a type (train, val or test) of set in the set_list
  as a tuple: first tuple element the directory of the fligth 
  and the second the number of images taken in that flight
  """
  set_list = []
  f = open(join(datasetdir, data_type+'_set.txt'), 'r')
  lst_runs = [ l.strip() for l in f.readlines() ]
  for run_dir in lst_runs:
    # print(run_dir)
    imgs_jpg=listdir(join(run_dir,'RGB'))
    # get list of all image numbers available in listdir
    num_imgs=sorted([int(im[0:-4]) for im in imgs_jpg])
    if not isfile(join(run_dir,'RGB','{0:010d}.jpg'.format(num_imgs[-1]))):
      print('ERROR:',run_dir,' imgnum: ',num_imgs[-1])
    control_file = open(join(run_dir,'control_info.txt'),'r')
    control_list = []
    ind = 0
    for ctr in control_file.readlines():
      control_ind = int(ctr.strip().split(' ')[0])
      control_val = float(ctr.strip().split(' ')[6])
      if ind<len(num_imgs) and num_imgs[ind]==control_ind:
        # clip at -1 and 1
        if abs(control_val) > 1: control_val = np.sign(control_val)
        control_list.append(control_val)
        ind+=1
        # print('added control ',control_val,' as target for ',control_ind)
    # cut the images for which no control is saved, should only be the case of the last frame
    num_imgs = num_imgs[:ind]
    assert len(num_imgs) == len(control_list), "Lenght of number of images {0} is not equal to number of control {1}".format(len(num_imgs),len(control_list))
    # Add depth links
    depth_list = [] 
    if FLAGS.auxiliary_depth:
      depths_jpg=listdir(join(run_dir,'Depth'))
      num_depths=sorted([int(de[0:-4]) for de in depths_jpg])
      smallest_depth = num_depths.pop(0)
      for ni in num_imgs: #link the indices of rgb images with the smallest depth bigger than current index
        while(ni > smallest_depth):
          try:
            smallest_depth = num_depths.pop(0)
          except IndexError:
            break
        depth_list.append(smallest_depth)
      num_imgs = num_imgs[:len(depth_list)]
      control_list = control_list[:len(depth_list)]
      assert len(num_imgs) == len(control_list) == len(depth_list), "Length of input(imags,control,depth) is not equal"
    set_list.append((run_dir, num_imgs, control_list, depth_list))
  f.close()
  # random.shuffle(set_list)
  return set_list

def prepare_data(size, size_depth=(55,74)):
  global im_size, full_set, de_size
  '''Load lists of tuples refering to images from which random batches can be drawn'''
  # stime = time.time()
  train_set = load_set('train')
  val_set=load_set('val')
  test_set=load_set('test')
  full_set={'train':train_set, 'val':val_set, 'test':test_set}
  im_size=size
  de_size = size_depth
  # print('duration: ',time.time()-stime)
  # import pdb; pdb.set_trace()
  
def generate_batch(data_type):
  """ Generator object that gets a random batch when next() is called
  yields: 
  - index: of current batch relative to the data seen in this epoch.
  - ok: boolean that defines if batch was loaded correctly
  - imb: batch of input rgb images
  - trgb: batch of corresponding control targets
  - auxb: batch with auxiliary info
  """
  data_set=full_set[data_type]
  number_of_frames = sum([len(run[1]) for run in data_set])
  # When there is that much data applied that you can get more than 100 minibatches out
  # stick to 100, otherwise one epoch takes too long and the training is not updated
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
      # im_b = np.zeros((FLAGS.batch_size,im_size[0],im_size[1],im_size[2])) #[] #list(range(FLAGS.batch_size)) #
      # trgt_b = np.zeros((FLAGS.batch_size, 1)) #[] #list(range(FLAGS.batch_size)) #
      # aux_b = np.zeros((FLAGS.batch_size,55,74))
      # im_b = []
      # trgt_b = []
      # aux_b =[]
      batch=[]
      # sample indices from dataset
      # from which threads can start loading
      batch_indices = []
      # checklist keeps track for each batch whether all the loaded data was loaded correctly
      checklist = []
      # list of samples to fill batch for threads to know what sample to load
      # thread_indices = list(range(FLAGS.batch_size))
      stime=time.time()
      for batch_num in range(FLAGS.batch_size):
        # choose random index over all runs:
        run_ind = random.choice(range(len(data_set)))
        # choose random index over image numbers:
        frame_ind = random.choice(range(len(data_set[run_ind][1])))
        batch_indices.append((batch_num, run_ind, frame_ind))
      # print("picking random indices duration: ",time.time()-stime)
      # import pdb; pdb.set_trace()
      def load_image_and_target(coord, batch_indices, batch, checklist):
        while not coord.should_stop():
          try:
            # loc_ind = thread_indices.pop()
            # print('------------',loc_ind)
            loc_ind, run_ind, frame_ind = batch_indices.pop()
            
            # Print debug info:
            # if FLAGS.auxiliary_depth:
            #   print("--- Run: {0}, im: {1}, ctrl: {2}, depth: {3}".format(data_set[run_ind][0],data_set[run_ind][1][frame_ind],
            #   data_set[run_ind][2][frame_ind],data_set[run_ind][3][frame_ind]))
            # else:
            #   print("--- Run: {0}, im: {1}, ctrl: {2}, depth: {3}".format(data_set[run_ind][0],data_set[run_ind][1][frame_ind],
            #   data_set[run_ind][2][frame_ind]))
            # get index of input image
            # run_ind, frame_ind = batch_indices[loc_ind]
            # load image
            img_file = join(data_set[run_ind][0],'RGB', '{0:010d}.jpg'.format(data_set[run_ind][1][frame_ind]))
            im = Image.open(img_file)
            im = sm.imresize(im,im_size,'nearest')
            # im = im * 1/255.
            # center the data around zero with 1standard devation
            # with tool 'get_mean_variance.py' in tensorflow2/examples/tools
            im -= FLAGS.mean
            im = im*1/FLAGS.std
            # im_b[loc_ind,:,:,:]=im
            # im_b[loc_ind]=im
            de = None          
            if FLAGS.auxiliary_depth:
              depth_file = join(data_set[run_ind][0],'Depth', '{0:010d}.jpg'.format(data_set[run_ind][3][frame_ind]))
              de = Image.open(depth_file)
              de = sm.imresize(de,de_size,'nearest')
              de = de * 1/255. * 5.
            # append rgb image, control and depth to batch
            batch.append((im, data_set[run_ind][2][frame_ind], de))
            # import pdb; pdb.set_trace()
            # # load target
            # control_file = open(join(data_set[run_ind][0],'control_info.txt'),'r')
            # # ! readlines() takes a long time...
            # control_list = []
            # for ctr in control_file.readlines():
            #   control_list.append([float(e) for e in ctr.strip().split(' ')])
            # control_ar=np.array(control_list)
            # # get the target yaw turn from the control and map between -1 and 1
            # #control = max(min(1, control_list[im_ind].split(' ')[6]), -1)
            # control = float(control_ar[control_ar[:,0]==im_ind,6][0])
            # print(control)
            # import pdb; pdb.set_trace()
            # # control = float(control_list[im_ind].split(' ')[6])
            # if abs(control) > 1: control = np.sign(control)
            # trgt_b[loc_ind, :]=[control]
            # #trgt_b.append([control])
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
        threads = [threading.Thread(target=load_image_and_target, args=(coord, batch_indices, batch, checklist)) for i in range(FLAGS.num_threads)]
        for t in threads: t.start()
        coord.join(threads, stop_grace_period_secs=5)
      except RuntimeError as e:
        print("threads are not stopping...",e)
      else:
        if len(checklist) != sum(checklist): ok=False
    if ok: b+=1
    yield b, ok, batch
    
#### FOR TESTING ONLY
if __name__ == '__main__':
  def print_dur(start_time):
    duration = (time.time()-start_time)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return "time: %dh:%02dm:%0.5fs" % (h, m, s)
  FLAGS.auxiliary_depth = True
  prepare_data((240,320,3))
  start_time=time.time()
  for index, ok, batch in generate_batch('train'):
    print('b: ',index,' ok ',ok,' ',print_dur(start_time))
    
    import pdb; pdb.set_trace()
    start_time=time.time()
  
    # if not ok: print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!problem...')#print(imb.shape, trgtb.shape)
    # #print imb.shape, trgtb.shape
    # print trgtb
    pass
  print('loading time one episode: ', print_dur(start_time))
  
