module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2
conda create --name pytorch
source activate pytorch
conda install pytorch torchvision -c pytorch
conda install matplotlib -c pytorch
conda env export > config/environment_pytorch.yml # save conda environment description
