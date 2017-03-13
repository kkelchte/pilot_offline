""" 
Inception trained in simulation supervised fashion
Author: Klaas Kelchtermans (based on code of Patrick Emami)
"""
#from lxml import etree as ET
import xml.etree.cElementTree as ET
import tensorflow as tf
import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.python.slim import model_analyzer as ma
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import random_ops

import rospy

import numpy as np
from model import Model
import data

import sys, os, os.path
import subprocess
import shutil
import time
import signal

FLAGS = tf.app.flags.FLAGS


# ===========================
#   Training Parameters
# ===========================
tf.app.flags.DEFINE_integer("max_episodes", 101, "The maximum number of episodes (~runs through all the training data.)")

# ===========================
#   Utility Parameters
# ===========================
# Print output of ros verbose or not
tf.app.flags.DEFINE_boolean("verbose", True, "Print output of ros verbose or not.")
# Directory for storing tensorboard summary results
tf.app.flags.DEFINE_string("summary_dir", '/home/klaas/tensorflow2/log/', "Choose the directory to which tensorflow should save the summaries.")
# Add log_tag to overcome overwriting of other log files
tf.app.flags.DEFINE_string("log_tag", 'testing', "Add log_tag to overcome overwriting of other log files.")
# Choose to run on gpu or cpu
tf.app.flags.DEFINE_string("device", '/gpu:0', "Choose to run on gpu or cpu: /cpu:0 or /gpu:0")
# Set the random seed to get similar examples
tf.app.flags.DEFINE_integer("random_seed", 123, "Set the random seed to get similar examples.")
# Overwrite existing logfolder
tf.app.flags.DEFINE_boolean("owr", False, "Overwrite existing logfolder when it is not testing.")
tf.app.flags.DEFINE_float("action_bound", 1.0, "Define between what bounds the actions can go. Default: [-1:1].")

# ===========================
#   Save settings
# ===========================
def save_config(logfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Save configuration in xml.")
  root = ET.Element("conf")
  flg = ET.SubElement(root, "flags")
  
  flags_dict = FLAGS.__dict__['__flags']
  for f in flags_dict:
    #print f, flags_dict[f]
    ET.SubElement(flg, f, name=f).text = str(flags_dict[f])
  tree = ET.ElementTree(root)
  tree.write(os.path.join(logfolder,file_name+".xml"), encoding="us-ascii", xml_declaration=True, method="xml")

def print_dur(start_time):
  duration = (time.time()-start_time)
  m, s = divmod(duration, 60)
  h, m = divmod(m, 60)
  return "time: %dh:%02dm:%02ds" % (h, m, s)

# Use the main method for starting the training procedure and closing it in the end.
def main(_):
  #Check log folders and if necessary remove:
  if FLAGS.log_tag == 'testing' or FLAGS.owr:
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag):
      shutil.rmtree(FLAGS.summary_dir+FLAGS.log_tag,ignore_errors=True)
  else :
    if os.path.isdir(FLAGS.summary_dir+FLAGS.log_tag):
      raise NameError( 'Logfolder already exists, overwriting alert: '+ FLAGS.summary_dir+FLAGS.log_tag ) 
  os.mkdir(FLAGS.summary_dir+FLAGS.log_tag)
  save_config(FLAGS.summary_dir+FLAGS.log_tag)

  # some startup settings
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
  
  #define the size of the image: SxS
  state_dim = inception.inception_v3.default_image_size
  action_dim = 1 #initially only turn and go straight
  
  print( "Number of State Dimensions:", state_dim)
  print( "Number of Action Dimensions:", action_dim)
  print( "Action bound:", FLAGS.action_bound)
  
  tf.logging.set_verbosity(tf.logging.DEBUG)
  
  # Random input from tensorflow (could be placeholder)
  images=random_ops.random_uniform((1,state_dim,state_dim,3))
  targets=random_ops.random_uniform((1,action_dim))
  
  config=tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  writer = tf.summary.FileWriter(FLAGS.summary_dir+FLAGS.log_tag, sess.graph)
  model = Model(sess, state_dim, action_dim, writer=writer, bound=FLAGS.action_bound)
  #model=None
  
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    #save checkpoint?
    print('saving checkpoints')
    model.save(0,FLAGS.summary_dir+FLAGS.log_tag)
    #rosinterface.close()
    sess.close()
    print('done.')
    sys.exit(0)
  print('------------Press Ctrl+C to end the learning')
  signal.signal(signal.SIGINT, signal_handler)
  
  #rosinterface.launch(FLAGS.launchfile)
  #rosnode = rosinterface.PilotNode(model, FLAGS.summary_dir+FLAGS.log_tag)
  data.prepare_data((state_dim, state_dim, 3))
  for ep in range(FLAGS.max_episodes):
    start_time=time.time()
    print('start episode:', ep)
    train_loss=[]
    val_loss=[]
    # train episode
    #for index, ok, im_b, trgt_b in data.generate_batch('train'):
      #if ok:
        #try:
          #_, batch_loss = model.backward(im_b, trgt_b)
        #except Exception as e:
          #print('failed to train on this batch: ',e)
        ##print(batch_loss)
        #train_loss.append(batch_loss)
    #if len(train_loss)==0:
      #raise IOError('Training failed on all batches of this episode.')  
    #print('training : ',print_dur(start_time),'loss:[avg {0:.3f}; max {1:.3f}; min {2:.3f}]'.format(np.mean(train_loss), max(train_loss), min(train_loss)))
    #sys.stdout.flush()
    train_loss=[1,2,3]
    
    # evaluate on val set
    val_time= time.time()
    activation_images=[]
    for index, ok, im_b, trgt_b in data.generate_batch('val'):
      if ok:
        _, batch_loss = model.backward(im_b, trgt_b)
        val_loss.append(batch_loss)
      if index == 1: 
	activation_images = model.plot_activations(im_b)
    print('validation : ',print_dur(val_time),'loss:[avg {0:.3f}; max {1:.3f}; min {2:.3f}]'.format(np.mean(val_loss), max(val_loss), min(val_loss)))
    sys.stdout.flush()
    # write summary
    try:
      sumvar=[np.mean(train_loss), np.mean(val_loss), activation_images]
      model.summarize(ep, sumvar)
    except Exception as e:
      print('failed to summarize', e)
    # write checkpoint every 100 episodes
    if (ep%20==0 and ep!=0) or ep==(FLAGS.max_episodes-1):
      print('saved checkpoint')
      model.save(ep,FLAGS.summary_dir+FLAGS.log_tag)
  # Training finished, time to test:
  # evaluate on val set
  test_loss=[]
  for ok, im_b, trgt_b in data.generate_batch('test'):
    if ok:
      _, batch_loss = model.backward(im_b, trgt_b)
      test_loss.append(batch_loss)
  print('validation : ',print_dur(start_time),'loss: [avg {0:.3f}; max {1:.3f}; min {2:.3f}]'.format(np.mean(test_loss), max(test_loss), min(test_loss)))
    
  #for i in range(10):
    #inpt, trgt = sess.run([images, targets])
    #action, loss = model.backward(inpt, trgt)
    #model.summarize(i, [loss, 10])
  #import pdb; pdb.set_trace()
    
  
    
if __name__ == '__main__':
  tf.app.run() 
