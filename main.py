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
import mobile_net
import depth_estim
import sys, os, os.path
import subprocess
import shutil
import time
import signal

# Block all the ugly printing...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


FLAGS = tf.app.flags.FLAGS

# ==========================
#   Training Parameters
# ==========================
tf.app.flags.DEFINE_integer("max_episodes", 100, "The maximum number of episodes (~runs through all the training data.)")

# ===========================
#   Utility Parameters
# ===========================
# Print output of ros verbose or not
tf.app.flags.DEFINE_boolean("verbose", True, "Print output of ros verbose or not.")
# Directory for storing tensorboard summary results
tf.app.flags.DEFINE_string("summary_dir", 'tensorflow/log/', "Choose the directory to which tensorflow should save the summaries.")
# tf.app.flags.DEFINE_string("summary_dir", '/esat/qayd/kkelchte/tensorflow/offline_log/', "Choose the directory to which tensorflow should save the summaries.")
# Add log_tag to overcome overwriting of other log files
tf.app.flags.DEFINE_string("log_tag", 'testing', "Add log_tag to overcome overwriting of other log files.")
# Choose to run on gpu or cpu
tf.app.flags.DEFINE_string("device", '/gpu:0', "Choose to run on gpu or cpu: /cpu:0 or /gpu:0")
# Set the random seed to get similar examples
tf.app.flags.DEFINE_integer("random_seed", 123, "Set the random seed to get similar examples.")
# Overwrite existing logfolder
tf.app.flags.DEFINE_boolean("owr", True, "Overwrite existing logfolder when it is not testing.")
tf.app.flags.DEFINE_float("action_bound", 1.0, "Define between what bounds the actions can go. Default: [-1:1].")
tf.app.flags.DEFINE_string("network", 'mobile_small', "Define the type of network: inception / depth / mobile.")
tf.app.flags.DEFINE_boolean("auxiliary_depth", False, "Specify whether a depth map is predicted.")

tf.app.flags.DEFINE_boolean("auxiliary_odom", False, "Specify whether the odometry or change in x,y,z,Y is predicted.")
tf.app.flags.DEFINE_boolean("plot_depth", False, "Specify whether the depth predictions is saved as images.")
tf.app.flags.DEFINE_boolean("lstm", False, "In case of True, cnn-features are fed into LSTM control layers.")
tf.app.flags.DEFINE_boolean("n_fc", False, "In case of True, prelogit features are concatenated before feeding to the fully connected layers.")
tf.app.flags.DEFINE_integer("n_frames", 3, "Specify the amount of frames concatenated in case of n_fc.")
tf.app.flags.DEFINE_integer("num_steps", 8, "Define the number of steps the LSTM layers are unrolled.")
tf.app.flags.DEFINE_integer("lstm_hiddensize", 100, "Define the number of hidden units in the LSTM control layer.")

tf.app.flags.DEFINE_boolean("rl", False, "In case of rl, use reinforcement learning to weight the gradients with a cost-to-go estimated from current depth.")

# ===========================
#   Save settings
# ===========================
def save_config(logfolder, file_name = "configuration"):
  """
  save all the FLAG values in a config file / xml file
  """
  print("Save configuration to: ", logfolder)
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
  summary_dir = os.path.join(os.getenv('HOME'),FLAGS.summary_dir)
  # summary_dir = FLAGS.summary_dir
  print("summary dir: {}".format(summary_dir))
  #Check log folders and if necessary remove:
  if FLAGS.log_tag == 'testing' or FLAGS.owr:
    if os.path.isdir(summary_dir+FLAGS.log_tag):
      shutil.rmtree(summary_dir+FLAGS.log_tag,ignore_errors=False)
  else :
    if os.path.isdir(summary_dir+FLAGS.log_tag):
      raise NameError( 'Logfolder already exists, overwriting alert: '+ summary_dir+FLAGS.log_tag ) 
  os.makedirs(summary_dir+FLAGS.log_tag) 
  # os.mkdir(summary_dir+FLAGS.log_tag)
  save_config(summary_dir+FLAGS.log_tag)
    
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
  elif FLAGS.network =='mobile':
    state_dim = [1, mobile_net.mobilenet_v1.default_image_size, mobile_net.mobilenet_v1.default_image_size, 3]  
  elif FLAGS.network =='mobile_small':
    state_dim = [1, mobile_net.mobilenet_v1.default_image_size_small, mobile_net.mobilenet_v1.default_image_size_small, 3]  
  else:
    raise NameError( 'Network is unknown: ', FLAGS.network)
    
  action_dim = 1 #initially only turn and go straight
  
  print( "Number of State Dimensions:", state_dim)
  print( "Number of Action Dimensions:", action_dim)
  print( "Action bound:", FLAGS.action_bound)
  # import pdb; pdb.set_trace()
  # tf.logging.set_verbosity(tf.logging.DEBUG)
  # inputs=random_ops.random_uniform(state_dim)
  # targets=random_ops.random_uniform((1,action_dim))
  # depth_targets=random_ops.random_uniform((1,1,1,64))
  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  model = Model(sess, state_dim, action_dim, bound=FLAGS.action_bound)
  writer = tf.summary.FileWriter(summary_dir+FLAGS.log_tag, sess.graph)
  model.writer = writer
  
  def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    #save checkpoint?
    print('saving checkpoints')
    model.save(summary_dir+FLAGS.log_tag)
    sess.close()
    print('done.')
    sys.exit(0)
  signal.signal(signal.SIGINT, signal_handler)
  print('------------Press Ctrl+C to end the learning') 
  
  def run_episode(data_type, sumvar):
    '''run over batches
    return different losses
    type: 'train', 'val' or 'test'
    '''
    activation_images = []
    depth_predictions = []
    endpoint_activations = []
    start_time=time.time()
    data_loading_time = 0
    calculation_time = 0
    start_data_time = time.time()
    tot_loss=[]
    ctr_loss=[]
    dep_loss=[]
    odo_loss=[]
    for index, ok, batch in data.generate_batch(data_type):
      data_loading_time+=(time.time()-start_data_time)
      start_calc_time=time.time()
      if ok:
        inputs = np.array([_['img'] for _ in batch])
        state = []
        targets = np.array([[_['ctr']] for _ in batch])
        target_depth = np.array([_['depth'] for _ in batch]).reshape(-1,55,74) if FLAGS.auxiliary_depth else []
        target_odom = np.array([_['odom'] for _ in batch]).reshape((-1,4)) if FLAGS.auxiliary_odom else []
        prev_action = np.array([_['prev_act'] for _ in batch]).reshape((-1,1)) if FLAGS.auxiliary_odom else []
        if data_type=='train':
          control, losses = model.backward(inputs, state, targets, depth_targets=target_depth, odom_targets=target_odom, prev_action=prev_action)
        elif data_type=='val' or data_type=='test':
          control, state, losses, aux_results = model.forward(inputs, state, auxdepth=False, auxodom=False, prev_action=prev_action, targets=targets, target_depth=target_depth, target_odom=target_odom)
        tot_loss.append(losses['t'])
        ctr_loss.append(losses['c'])
        if FLAGS.auxiliary_depth: dep_loss.append(losses['d'])
        if FLAGS.auxiliary_odom: odo_loss.append(losses['o'])
        if index == 1 and data_type=='val':
          if FLAGS.plot_activations:
            activation_images = model.plot_activations(inputs, targets)
          if FLAGS.plot_depth:
            depth_predictions = model.plot_depth(inputs, target_depth)
          if FLAGS.plot_histograms:
            # stime = time.time()
            endpoint_activations = model.get_endpoint_activations(inputs)
            # print('plot activations: {}'.format((stime-time.time())))
      calculation_time+=(time.time()-start_calc_time)
      start_data_time = time.time()
    print('>>{0} [{1[2]}/{1[1]}_{1[3]:02d}:{1[4]:02d}]: data {2}; calc {3}'.format(data_type.upper(),tuple(time.localtime()[0:5]),
      print_dur(data_loading_time),print_dur(calculation_time)))
    print('losses: tot {0:.3g}; ctrl {1:.3g}; depth {2:.3g}; odom {2:.3g};'.format(np.mean(tot_loss), np.mean(ctr_loss), np.mean(dep_loss), np.mean(odo_loss)))
    sys.stdout.flush()
    sumvar['loss_total_'+data_type]=np.mean(tot_loss)
    sumvar['loss_control_'+data_type]=np.mean(ctr_loss)
    if FLAGS.auxiliary_depth: sumvar['loss_depth_'+data_type]=np.mean(dep_loss)
    if FLAGS.auxiliary_odom: sumvar['loss_odom_'+data_type]=np.mean(odo_loss)
    
    if len(activation_images) != 0:
      sumvar['conv_activations']=activation_images
    if len(depth_predictions) != 0:
      sumvar['depth_predictions']=depth_predictions
    if FLAGS.plot_histograms:
      for i, ep in enumerate(model.endpoints):
        sumvar['activations_{}'.format(ep)]=endpoint_activations[i]
    return sumvar

  data.prepare_data((state_dim[1], state_dim[2], state_dim[3]))
  for ep in range(FLAGS.max_episodes):
    print('start episode: {}'.format(ep))
    # ----------- train episode
    sumvar = run_episode('train', {})
    
    # ----------- validate episode
    # sumvar = run_episode('val', {})
    sumvar = run_episode('val', sumvar)
    # import pdb; pdb.set_trace()
  
    # ----------- write summary
    try:
      model.summarize(sumvar)
    except Exception as e:
      print('failed to summarize {}'.format(e))
    # write checkpoint every x episodes
    if (ep%20==0 and ep!=0):
      print('saved checkpoint')
      model.save(summary_dir+FLAGS.log_tag)
  # ------------ test
  sumvar = run_episode('test', {})  
  # ----------- write summary
  try:
    model.summarize(sumvar)
  except Exception as e:
    print('failed to summarize {}'.format(e))
  # write checkpoint every x episodes
  if (ep%20==0 and ep!=0) or ep==(FLAGS.max_episodes-1):
    print('saved checkpoint')
    model.save(summary_dir+FLAGS.log_tag)
  
    
if __name__ == '__main__':
  tf.app.run() 
