# This code use mvit for DGCNN-based embedding classification 
Build the environment following [INSTALL.md](https://github.com/Z-702/MVIT_classification/blob/main/INSTALL.md) 
# Using DGCNN to generate embedding images
Usng following pipeline, [embed.py](https://github.com/Z-702/MVIT_classification/blob/main/embed.py) to generate embedding images from DTI inputs.  
The used codes are shown in the folder, embed_tools.  
Some of the codes originated from the DeepWMA folder; they could be found on each server. Thus, I think there is no need to put them here. 
# Multiple inputs such as FA, MD and Density are used to train and evaluate the model
1. For different datasets, you need to change line 164 and line 169 of ./mvit/dataset/tractoembedding.py
2. For model setting, edit config set in ./config/MVOTv2_mri.yaml
# Edit the modility you use
1. Change line 146 of ./mvit/dataset/tractoembedding.py,
```bash
for mode in ['FA1', 'density', 'trace1']:
```
2. Change parapmeter IN_CHANS in ./config/MVOTv2_mri.yaml to determine the number of modility 
# Execute code as follows
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$(pwd) \
nohup python ./tools/main.py \
    --cfg configs/MVITv2_mri.yaml \
    DATA.PATH_TO_DATA_DIR fold.csv \
    NUM_GPUS 2 \
    TRAIN.BATCH_SIZE 10 \
    OUTPUT_DIR path\to\your\model \
    DATA_NUM {num} \
    DATA_AUG_NUM {N} \
    > output_new500_v2.log 2>&1 &
```
The csv file is supposed to be coded with three columns, SUB_ID, DX_GROUP and fold. The program is supposed multi-fold evaluation, and I suggest to use five-fold cross evaluation.  
DATA_NUM is the utilized embedding locations (1 for left, 2 for left and right, 3 for left, right and commisure)  
DATA_AUG_NUM is the utilized augumentation amount 
# The data path should be named following this
/data01/zixi/HCP_500_vtk/129634/tractoembedding/da-full/129634-trace1_CLR_sz640.nii.gz 
1. /data01/zixi/HCP_500_vtk/129634/tractoembedding: root path 
2. /129634: subject id
3. /da-full: augumentation index
4. trace1: modility
5. sz640: resolution(default utilizing three resolution:['sz80', 'sz160', 'sz320'], can be changed on line 64 of ./mvit/dataset/tractpembedding.py


