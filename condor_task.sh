#!/usr/bin/bash
# This scripts sets some parameters for running a tasks,
# creates a condor and shell scripts and launches the stuff on condor.

TASK='main.py --max_episodes 1000 --network depth --dataset mix --model_path depth_net_checkpoint --auxiliary_depth True' # 0 	
#1: normal
log_tag="$(date +%F_%H%M)_mix_auxdepth"
#2: with gradient multipliers
#grad_mul="True"
#learning_rate="0.001"


condor_output_dir='/esat/qayd/kkelchte/tensorflow/condor_output'
#------------------------------------
if [ ! -z $model ]; then 
	TASK="$TASK --model ${model}"
	description="${description}_$model"
fi
if [ ! -z $network ]; then 
	TASK="$TASK --network ${network}"
	description="${description}_net_$network"
fi
if [ ! -z $wsize ]; then 
	TASK="$TASK --window_size ${wsize}"
	description="${description}_ws_$wsize"
fi
if [ ! -z $batchsize ]; then 
	TASK="$TASK --batch_size ${batchsize}"
	description="${description}_bs_$batchsize"
fi
if [ ! -z $sample ]; then 
	TASK="$TASK --sample ${sample}"
	description="${description}_sample_${sample}"
fi
if [ ! -z $log_tag ]; then 
	TASK="$TASK --log_tag ${log_tag}"
	description="${description}_${log_tag}"
fi
if [ ! -z $model_path ]; then
    TASK="$TASK --model_path ${model_path}"
fi
if [ ! -z $checkpoint_path ]; then
    TASK="$TASK --checkpoint_path ${checkpoint_path}"
    description="${description}_ckpt"
fi
if [ ! -z $learning_rate ]; then 
	TASK="$TASK --learning_rate ${learning_rate}"
	description="${description}_lr_$learning_rate"
fi
if [ ! -z $grad_mul ]; then 
	TASK="$TASK --grad_mul ${grad_mul}"
	description="${description}_gradmul"
fi
if [ ! -z ${optimizer} ]; then 
	TASK="$TASK --optimizer ${optimizer}"
	description="${description}_opt_${optimizer}"
fi
if [ ! -z $normalized ]; then 
	TASK="$TASK --normalized ${normalized}"
	description="${description}_norm_$normalized"
fi
if [ ! -z $random_order ]; then 
	TASK="$TASK --random_order ${random_order}"
	description="${description}_rand_$random_order"
fi
if [ ! -z $dataset ]; then 
	TASK="$TASK --dataset ${dataset}"
	#description="${description}_rand_$random_order"
fi

echo $TASK
# Delete previous log files if they are there
if [ -d $condor_output_dir ];then
rm -f "$condor_output_dir/condor${description}.log"
rm -f "$condor_output_dir/condor${description}.out"
rm -f "$condor_output_dir/condor${description}.err"
else
mkdir $condor_output_dir
fi
temp_dir="/users/visics/kkelchte/tensorflow/examples/pilot_offline/.tmp"
condor_file="${temp_dir}/condor${description}.condor"
shell_file="${temp_dir}/run${description}.sh"
prescript_file="${temp_dir}/prescript${description}.sh"
mkdir -p $temp_dir
#--------------------------------------------------------------------------------------------
echo "Universe         = vanilla" > $condor_file
echo "">> $condor_file
echo "RequestCpus      = 4" >> $condor_file
echo "Request_GPUs = 1" >> $condor_file

# ---4g option---
if [ -z $andromeda ]; then
echo "RequestMemory = 15900" >> $condor_file
echo "Requirements = (CUDAGlobalMemoryMb >= 1900) && (CUDACapability >= 3.5)">> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 3900) && (CUDACapability >= 3.5)">> $condor_file
# echo "Requirements = (CUDAGlobalMemoryMb >= 3900) && (CUDACapability >= 3.5) && (machine != \"askoy.esat.kuleuven.be\" )">> $condor_file
else
# ---andromeda option---
echo "RequestMemory = 62000" >> $condor_file
echo "Requirements = (CUDAGlobalMemoryMb >= 3900) && (CUDACapability >= 3.5)">> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 11000) && (CUDACapability >= 3.5)">> $condor_file
fi
# ---Trash option---
#echo "Requirements = (CUDAGlobalMemoryMb >= 3900) && (CUDACapability >= 3.5) && (CUDADeviceName == \"GeForce GTX 960\" || CUDADeviceName == \"GeForce GTX 980\" )">> $condor_file

#echo "RequestMemory = 16G" >> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 1900) && (CUDACapability >= 3.5) && (machineowner == \"Visics\") && (machine != \"amethyst.esat.kuleuven.be\" ) && (CUDADeviceName == 'GeForce GTX 960' || CUDADeviceName == 'GeForce GTX 980' )" >> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 1900) && (CUDACapability >= 3.5)" >> $condor_file

echo "RequestDisk      = 25G" >> $condor_file
#wall time ==> generally assumed a job should take 6hours longest,
#if you want longer or shorter you can set the number of seconds. (max 1h ~ +3600s)
#100 hours means 4 days 
# echo "+RequestWalltime = 360000" >> $condor_file 
#echo "+RequestWalltime = 10800" >> $condor_file
# echo "">> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb > 1900) && (CUDADeviceName == 'GeForce GTX 960' || CUDADeviceName == 'GeForce GTX 980' ) && (machineowner == Visics)" >> $condor_file
#echo "Requirements = ((machine == "vega.esat.kuleuven.be") || (machine == "wasat.esat.kuleuven.be") || (machine == "yildun.esat.kuleuven.be"))" >> $condor_file

#echo "Requirements = ((machine != \"izar.esat.kuleuven.be\") && (machine != \"oculus.esat.kuleuven.be\")  && (machine != \"emerald.esat.kuleuven.be\"))" >> $condor_file
echo "Niceuser = true" >> $condor_file

echo "Initialdir   = $temp_dir" >> $condor_file
echo "Executable   = $shell_file" >> $condor_file
#echo "+PreCmd      = \"$prescript_file\"" >> $condor_file
echo "Log 	   = $condor_output_dir/condor${description}.log" >> $condor_file
echo "Output       = $condor_output_dir/condor${description}.out" >> $condor_file
echo "Error        = $condor_output_dir/condor${description}.err" >> $condor_file
echo "">> $condor_file
#mail kkelchte on Error or Always
echo "Notification = Error" >> $condor_file
echo "Queue" >> $condor_file

echo "#!/usr/bin/bash" > $shell_file
echo "task='"${TASK}"'">>$shell_file
echo 'echo $task'>>$shell_file
echo "##-------------------------------------------- ">>$shell_file
echo "echo 'run_the_thing has started' ">>$shell_file
echo "# load cuda and cdnn path in load library path">>$shell_file
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/users/visics/kkelchte/local/lib/cudnn-5.1/cuda/lib64:">>$shell_file
echo "# run virtual environment for tensorflow">>$shell_file
echo "source /users/visics/kkelchte/tensorflow/bin/activate">>$shell_file
echo "# set python library path">>$shell_file
echo "export PYTHONPATH=/users/visics/kkelchte/tensorflow/lib/python2.7/site-packages:/users/visics/kkelchte/tensorflow/examples">>$shell_file
echo "ulimit -v unlimited">>$shell_file
echo "cd /users/visics/kkelchte/tensorflow/examples/pilot_offline">>$shell_file
echo "echo 'went to directory ' $PWD">>$shell_file
echo "python $TASK">>$shell_file
echo "echo '$TASK has finished. description: $description. $condor_file' | mailx -s 'condor' klaas.kelchtermans@esat.kuleuven.be">>$shell_file
#--------------------------------------------------------------------------------------------
#echo "stamp=\$( date +\"%F-%T\" )">>$prescript_file
#echo "if [ -e $condor_output_dir/condor${description}.log ] ; then mv $condor_output_dir/condor${description}.log $condor_output_dir/condor${description}_"'$stamp'".log; fi " >>$prescript_file
#echo "if [ -e $condor_output_dir/condor${description}.err ] ; then mv $condor_output_dir/condor${description}.err $condor_output_dir/condor${description}_"'$stamp'".err; fi " >>$prescript_file
#echo "if [ -e $condor_output_dir/condor${description}.out ] ; then mv $condor_output_dir/condor${description}.out $condor_output_dir/condor${description}_"'$stamp'".out; fi " >>$prescript_file
#--------------------------------------------------------------------------------------------

chmod 755 $condor_file
chmod 755 $shell_file
#chmod 755 $prescript_file

condor_submit $condor_file
echo $condor_file
