import os
import numpy as np
import tensorflow as tf
import threading
from os import listdir
from os.path import isfile, join, isdir
import time
import random
import h5py
from math import floor

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
tf.app.flags.DEFINE_string("dataset", "esat","pick the dataset in data_root from which your movies can be found.")
tf.app.flags.DEFINE_integer("batch_size", 16, "The size of the minibatch used for training.")
tf.app.flags.DEFINE_string("data_root", "/esat/qayd/kkelchte/docker_home/pilot_data", "Define the root folder of the different datasets.")
tf.app.flags.DEFINE_integer("num_threads", 4, "The number of threads for loading one minibatch.")
tf.app.flags.DEFINE_float("mean", 0, "Define the mean of the input data for centering around zero. Esat data:0.2623")
tf.app.flags.DEFINE_float("std", 1, "Define the standard deviation of the data for normalization. Esat data:0.1565")
tf.app.flags.DEFINE_boolean("joint_training", False, "Train the offline control jointly with the esat depth dataset.")
tf.app.flags.DEFINE_string("esat_data_file", "/esat/qayd/kkelchte/pilot_data/esat_real_depth.hdf5", "The filename of the hdf5 esat dataset.")

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
    assert len(num_imgs)!=0 , IOError('no images in {0}: {1}'.format(run_dir,len(imgs_jpg)))
    if not isfile(join(run_dir,'RGB','{0:010d}.jpg'.format(num_imgs[-1]))):
      print('ERROR:',run_dir,' imgnum: ',num_imgs[-1])
    # parse control data  
    control_file = open(join(run_dir,'control_info.txt'),'r')
    control_file_list = control_file.readlines()
    while len(control_file_list[-1])<=1 : control_file_list=control_file_list[:-1]
    # print(run_dir)
    control_parsed = [(int(ctr.strip().split(' ')[0]),float(ctr.strip().split(' ')[6])) for ctr in control_file_list]
    # import pdb; pdb.set_trace()
    def sync_control():
      control_list = []
      corresponding_imgs = []
      ctr_ind, ctr_val = control_parsed.pop(0)
      for ni in num_imgs:
        # print("ni: {}".format(ni))
        while(ctr_ind < ni):
          try:
            ctr_ind, ctr_val = control_parsed.pop(0)
            # print("ctr_ind: {}".format(ctr_ind))
          except IndexError:
            # print("return corresponding_imgs: {} \n control_list{}".format(corresponding_imgs, control_list))
            return corresponding_imgs, control_list
        # clip at -1 and 1
        if abs(ctr_val) > 1: ctr_val = np.sign(ctr_val)
        control_list.append(ctr_val)
        corresponding_imgs.append(ni)
        # print("append img {}: ctr num {}".format(ni, ctr_ind))
      return corresponding_imgs, control_list
    num_imgs, control_list = sync_control()
    assert len(num_imgs) == len(control_list), "Length of number of images {0} is not equal to number of control {1}".format(len(num_imgs),len(control_list))
    
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
  global im_size, full_set, de_size, esat_depth_file, esat_key, max_key
  '''Load lists of tuples refering to images from which random batches can be drawn'''
  # stime = time.time()
  train_set = load_set('train')
  val_set=load_set('val')
  test_set=load_set('test')
  full_set={'train':train_set, 'val':val_set, 'test':test_set}
  im_size=size
  de_size = size_depth
  if FLAGS.joint_training:
    esat_depth_file=h5py.File(FLAGS.esat_data_file, 'r')
    max_key = int(floor(esat_depth_file["depth"]["depth_data"].shape[2]/FLAGS.batch_size))
    esat_key = 0
  # print('duration: ',time.time()-stime)
  # import pdb; pdb.set_trace()
  
def generate_batch(data_type):
  global esat_key
  """ 
  input:
    data_type: 'train', 'val' or 'test'
  Generator object that gets a random batch when next() is called
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
      print('batch {0} of {1}'.format(b,number_of_batches))
    if b>0 and b%5==0 and FLAGS.joint_training:
      load_esat_depth = True
    else:
      load_esat_depth = False
    #print('batch ',cnt,' of ',number_of_batches)
    ok = True
    
    if load_esat_depth:
      print('-esat_depth_batch-')
      start_i = esat_key*FLAGS.batch_size
      end_i = (esat_key+1)*FLAGS.batch_size
      im_batch = np.rollaxis(esat_depth_file["rgb"]["rgb_data"][:,:,:,start_i:end_i],3)
      de_batch = np.rollaxis(esat_depth_file["depth"]["depth_data"][:,:,start_i:end_i],2)
      batch = [(im_batch[i], 999, de_batch[i]) for i in range(FLAGS.batch_size)]
      esat_key += 1
      if esat_key >= max_key:
        esat_key = 0
      # import pdb; pdb.set_trace()
    else: 
      #Multithreaded implementation
      # sample indices from dataset
      # from which threads can start loading
      batch=[]
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
        if FLAGS.n_fc:
          frame_ind = random.choice(range(len(data_set[run_ind][1])-FLAGS.n_frames))
        batch_indices.append((batch_num, run_ind, frame_ind))
      # print("picking random indices duration: ",time.time()-stime)
      # import pdb; pdb.set_trace()
      def load_image_and_target(coord, batch_indices, batch, checklist):
        while not coord.should_stop():
          try:
            # print('------------',loc_ind)
            loc_ind, run_ind, frame_ind = batch_indices.pop()
            def load_rgb_depth_image(run_ind, frame_ind):
              # load image
              img_file = join(data_set[run_ind][0],'RGB', '{0:010d}.jpg'.format(data_set[run_ind][1][frame_ind]))
              img = Image.open(img_file)
              img = sm.imresize(img,im_size,'nearest').astype(float) #.astype(np.float32)
              # center the data around zero with 1standard devation
              # with tool 'get_mean_variance.py' in tensorflow2/examples/tools
              img -= FLAGS.mean
              img = img*1/FLAGS.std
              
              de = None          
              if FLAGS.auxiliary_depth:
                depth_file = join(data_set[run_ind][0],'Depth', '{0:010d}.jpg'.format(data_set[run_ind][3][frame_ind]))
                de = Image.open(depth_file)
                de = sm.imresize(de,de_size,'nearest')
                de = de * 1/255. * 5.
              return img, de
            
            if FLAGS.n_fc:
              ims = []
              for frame in range(FLAGS.n_frames):
                image, de = load_rgb_depth_image(run_ind, frame_ind+frame) # target depth (de) is each time overwritten
                ims.append(image)
              im = np.concatenate(ims, axis=2)
            else:
              im, de = load_rgb_depth_image(run_ind, frame_ind)
            # append rgb image, control and depth to batch
            batch.append((im, data_set[run_ind][2][frame_ind], de))
            
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
def print_dur(start_time):
    duration = (time.time()-start_time)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return "time: %dh:%02dm:%0.5fs" % (h, m, s)
  
if __name__ == '__main__':
  FLAGS.auxiliary_depth = False
  prepare_data((240,320,3))

  # import pdb; pdb.set_trace()
  print 'run_dir:'
  full_set['train'][0][0]
  print 'len images: {}'.format(len(full_set['train'][0][1]))
  print 'len control: {}'.format(len(full_set['train'][0][2]))
  print 'len depth: {}'.format(len(full_set['train'][0][3]))
  
  # start_time=time.time()
  # for index, ok, batch in generate_batch('train'):
  #   print('b: ',index,' ok ',ok,' ',print_dur(start_time))
  #   print 'RGB Image:'
  #   print batch[0][0].shape
  #   print type(batch[0][0][0,0,0])
  #   print 'min: ',np.amin(batch[0][0]),' max: ',np.amax(batch[0][0])
  #   print 'depth Image:'
  #   print batch[0][2].shape
  #   print type(batch[0][2][0,0])
  #   print 'min: ',np.amin(batch[0][2]),' max: ',np.amax(batch[0][2])
  #   import pdb; pdb.set_trace()
  #   start_time=time.time()

  #   pass
  # print('loading time one episode: ', print_dur(start_time))
  
