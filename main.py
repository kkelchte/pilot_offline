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

import numpy as np
from model import Model
import data

import sys, os, os.path
import subprocess
import shutil
import time
import signal

import depth_estim

FLAGS = tf.app.flags.FLAGS


# ===========================
#   Training Parameters
# ===========================
tf.app.flags.DEFINE_integer("max_episodes", 1001, "The maximum number of episodes (~runs through all the training data.)")

# ===========================
#   Utility Parameters
# ===========================
# Print output of ros verbose or not
tf.app.flags.DEFINE_boolean("verbose", True, "Print output of ros verbose or not.")
# Directory for storing tensorboard summary results
tf.app.flags.DEFINE_string("summary_dir", '/esat/qayd/kkelchte/tensorflow/offline_log/', "Choose the directory to which tensorflow should save the summaries.")
# Add log_tag to overcome overwriting of other log files
tf.app.flags.DEFINE_string("log_tag", 'testing', "Add log_tag to overcome overwriting of other log files.")
# Choose to run on gpu or cpu
tf.app.flags.DEFINE_string("device", '/gpu:0', "Choose to run on gpu or cpu: /cpu:0 or /gpu:0")
# Set the random seed to get similar examples
tf.app.flags.DEFINE_integer("random_seed", 123, "Set the random seed to get similar examples.")
# Overwrite existing logfolder
tf.app.flags.DEFINE_boolean("owr", False, "Overwrite existing logfolder when it is not testing.")
tf.app.flags.DEFINE_float("action_bound", 1.0, "Define between what bounds the actions can go. Default: [-1:1].")

tf.app.flags.DEFINE_string("network", 'inception', "Define the type of network: inception / depth.")
tf.app.flags.DEFINE_boolean("auxiliary_depth", False, "Specify whether the horizontal line of depth is predicted as auxiliary task in the feature.")
tf.app.flags.DEFINE_boolean("plot_depth", False, "Specify whether the depth predictions is saved as images.")

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

def print_dur(duration_time):
  duration = duration_time #(time.time()-start_time)
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
  
  #define the size of the network input 
  if FLAGS.network == 'inception':
    state_dim = [1, inception.inception_v3.default_image_size, inception.inception_v3.default_image_size, 3]
  elif FLAGS.network == 'fc_control':
    state_dim = [1, fc_control.fc_control_v1.input_size]
  elif FLAGS.network =='depth':
    state_dim = depth_estim.depth_estim_v1.input_size
  else:
    raise NameError( 'Network is unknown: ', FLAGS.network)
  # state_dim = inception.inception_v3.default_image_size
  action_dim = 1 #initially only turn and go straight
  
  print( "Number of State Dimensions:", state_dim)
  print( "Number of Action Dimensions:", action_dim)
  print( "Action bound:", FLAGS.action_bound)
  
  tf.logging.set_verbosity(tf.logging.DEBUG)
  
  # inputs=random_ops.random_uniform(state_dim)
  # targets=random_ops.random_uniform((1,action_dim))
  # depth_targets=random_ops.random_uniform((1,1,1,64))
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  writer = tf.summary.FileWriter(FLAGS.summary_dir+FLAGS.log_tag, sess.graph)
  model = Model(sess, state_dim, action_dim, writer=writer, bound=FLAGS.action_bound)
  #model=None
  
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    #save checkpoint?
    print('saving checkpoints')
    model.save(FLAGS.summary_dir+FLAGS.log_tag)
    #rosinterface.close()
    sess.close()
    print('done.')
    sys.exit(0)
  print('------------Press Ctrl+C to end the learning')
  signal.signal(signal.SIGINT, signal_handler)
  
  def run_episode(data_type):
    '''run over batches
    return different losses
    type: 'train', 'val' or 'test'
    '''
    activation_images = []
    depth_predictions = []
    start_time=time.time()
    data_loading_time = 0
    calculation_time = 0
    start_data_time = time.time()
    tot_loss=[]
    ctr_loss=[]
    dep_loss=[]
    for index, ok, batch in data.generate_batch(data_type):
      data_loading_time+=(time.time()-start_data_time)
      start_calc_time=time.time()
      if ok:
        im_b = np.array([_[0] for _ in batch])
        trgt_b = np.array([[_[1]] for _ in batch])
        depth_b = np.array([_[2] for _ in batch])
        if not FLAGS.auxiliary_depth:
          if data_type=='train':
            _, losses = model.backward(im_b, trgt_b)
          else:
            _, losses = model.forward(im_b, targets = trgt_b)
          dep_loss.append(0)
        else:
          if data_type=='train':
            _, losses = model.backward(im_b, trgt_b, depth_b)
          else:
            _, losses = model.forward(im_b, targets = trgt_b, depth_targets = depth_b)
          dep_loss.append(losses[2])
        tot_loss.append(losses[0])
        ctr_loss.append(losses[1])
        if index == 1 and data_type=='val':
          if FLAGS.save_activations:
            activation_images = model.plot_activations(im_b)
          if FLAGS.plot_depth:
            depth_predictions = model.plot_depth(im_b, depth_b)
      calculation_time+=(time.time()-start_calc_time)
      start_data_time = time.time()
    if len(tot_loss)==0:
      raise IOError('Running episode ',data_type,' failed on all batches of this episode.')  
    print('>>{0}: data time {1}; calc time {2}'.format(data_type.upper(),print_dur(data_loading_time),print_dur(calculation_time)))
    print('losses: tot {0:.3g}; ctrl {1:.3g}; depth {2:.3g}'.format(np.mean(tot_loss), np.mean(ctr_loss), np.mean(dep_loss)))
    sys.stdout.flush()

    results = [np.mean(tot_loss), np.mean(ctr_loss), np.mean(dep_loss)]
    if len(activation_images) != 0:
      results.append(activation_images)
    if len(depth_predictions) != 0:
      results.append(depth_predictions)
    return results

  data.prepare_data((state_dim[1], state_dim[2], state_dim[3]))
  # import pdb; pdb.set_trace()
  for ep in range(FLAGS.max_episodes):
    print('start episode:', ep)
    sumvar=[]
    # ----------- train episode
    results = run_episode('train')
    sumvar.extend(results)
    
    # ----------- validate episode
    results = run_episode('val')
    sumvar.extend(results)
      
    # ----------- write summary
    try:
      model.summarize(sumvar)
    except Exception as e:
      print('failed to summarize', e)
    # write checkpoint every x episodes
    if (ep%20==0 and ep!=0) or ep==(FLAGS.max_episodes-1):
      print('saved checkpoint')
      model.save(FLAGS.summary_dir+FLAGS.log_tag)
  # ------------ test
  results = run_episode('test')  
  
    
if __name__ == '__main__':
  tf.app.run() 
