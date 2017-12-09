module load cudnn/v6.0-prod
module load python3/3.5.1
module load scipy/scipy-0.17.1-python-3.5.1
module load matplotlib/matplotlib-1.5.3-python-3.5.1
source /appl/tensorflow/1.3gpu-python3.5/bin/activate
/appl/glibc/2.17/lib/ld-linux-x86-64.so.2  --library-path /appl/glibc/2.17/lib/:/appl/gcc/4.8.5/lib64/:/usr/lib64/atlas:/lib64:/usr/lib64:$LD_LIBRARY_PATH $(which python)
