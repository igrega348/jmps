#!/bin/bash
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J jmps
#SBATCH -A DESHPANDE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:20:00
#SBATCH --mail-type=ALL
##SBATCH --output=./slurm_logs/slurm-%j.log

#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! Notes:
#! Charging is determined by GPU number*walltime. 

JOBID=$SLURM_JOB_ID

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load cudnn/8.2.4.15-11.4/gcc-9.4.0-vuxt37p
module load miniconda/3
echo "Activating conda environment"
source /usr/local/software/master/miniconda/3/bin/activate ~/graphgym-env/


#! Work directory (i.e. where the job will run):
# workdir=$(realpath `pwd`)
workdir=$(realpath scripts)

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

export RUNNAME=88
CMD="python train_fromcfg.py"


cd $workdir
echo -e "Changed directory to `pwd`.\n"


echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
start=`date +%s`
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Machine file:
        export NODEFILE=`generate_pbs_nodefile`
        echo -e "\nNodes allocated:\n================"
        echo $(cat $NODEFILE | uniq)
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 

end=`date +%s`
t=$(($end-$start))
h=$(($t/3600))
m=$(($t/60-$h*60))
s=$(($t-$h*3600-$m*60))
echo "Total walltime $h h : $m m : $s s"
