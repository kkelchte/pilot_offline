# pilot_offline
Tensorflow 1.1 code for training DNN policy from an offline dataset. It is used in the [DoShiCo challenge](https://kkelchte.github.io/doshico).

## Dependencies
* [Tensorflow (>1.1)](https://www.tensorflow.org/install/) or [docker image](https://hub.docker.com/r/kkelchte/ros_gazebo_tensorflow/) up and running.
* [data]("https://homes.esat.kuleuven.be/~kkelchte/pilot_data/data.zip"): this zip file contains the offline datasets:
  * Training data: collected in the simulated environments: [canyon, forest and sandbox](https://homes.esat.kuleuven.be/~kkelchte/data/pilot_data/canyon_forest_sandbox.zip)
  * Validation data: collected in the simulated environment: [ESAT](https://homes.esat.kuleuven.be/~kkelchte/data/pilot_data/esat.zip)
  * Test data: collected in the real world: [Almost-Collision Dataset](https://homes.esat.kuleuven.be/~kkelchte/data/pilot_data/almost_collision_set.zip)
* [log]("https://homes.esat.kuleuven.be/~kkelchte/checkpoints/models.zip"): this directory contains checkpoints of trained models, required to reproduce results.


## Installation
You can use this code from within the [docker image](https://hub.docker.com/r/kkelchte/ros_gazebo_tensorflow/) I supply for the [Doshico challenge](http://kkelchte.github.io/doshico).
```bash
$ git clone https://www.github.com/kkelchte/pilot_offline
# within a running docker container or tensorflow-virtual environment
$$  python main.py
```
In order to make it work, you can either adjust some default flag values or adapt the same folder structure.
* summary_dir (main.py): log folder to keep checkpoints and log info: $HOME/tensorflow/log
* checkpoint_path (model.py): [log folder]("https://homes.esat.kuleuven.be/~kkelchte/checkpoints/offl_mobsm_test.zip") from which checkpoints of models are read from: $HOME/tensorflow/log
* data_root (data.py): the folder in which the [data]("https://homes.esat.kuleuven.be/~kkelchte/pilot_data/data.zip") is saved: $HOME/pilot_data

It is best to download the log folder and save it on the correct relative path as well as the data folder.
