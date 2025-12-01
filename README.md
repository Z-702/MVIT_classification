# This code ues mvit for DGCNN based embedding classification 
Build the environment following [INSTALL.md](https://github.com/Z-702/MVIT_classification/blob/main/INSTALL.md) 
# Multiple inputs such as FA, MD and density are used to train and evaluate the model
1. For different datasets, you need to change line 164 and line 169 of ./mvit/dataset/tractoembedding.py
2. For model setting, edit config set in ./config/MVOTv2_mri.yaml
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
DATA_NUM is the utilized embedding locations (1 for left, 2 for left and righ, 3 for left, right and commisure)
DATA_AUG_NUM is the utilized augumentation amount
# The 


