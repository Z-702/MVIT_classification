import subprocess
import pathlib
from multiprocessing import Pool
import os
os.environ['LD_LIBRARY_PATH'] = ''
MODE = 'fusion'
N = 3
def run(num,device):
    subprocess.run(f'CUDA_VISIBLE_DEVICES={device} \
                   /home/junyi/.conda/envs/mvit/bin/python tools/main.py \
                   --cfg configs/MVITv2_mri.yaml \
                   DATA.PATH_TO_DATA_DIR fold.csv \
                   NUM_GPUS 1 \
                   TRAIN.BATCH_SIZE 16 \
                   OUTPUT_DIR /data06/junyi/results/mvit_direct_voting_weighted_new/{MODE}_{N}_{num:03d}\
                   DATA_MODE {MODE} \
                   DATA_AUG_NUM {num} \
                   DATA_NUM {N}',shell=True)

# for num in Num_lst:
#     for aug_num in Aug_num_lst:
#         for mode in mode_lst:
#             # if not pathlib.Path(f'/data01/junyi/results/mvit/{mode}_{num}_{aug_num}_').exists():
#             subprocess.run(f'CUDA_VISIBLE_DEVICES=7 /home/junyi/.conda/envs/mvit/bin/python tools/main.py --cfg configs/MVITv2_mri.yaml DATA.PATH_TO_DATA_DIR new.csv NUM_GPUS 1 TRAIN.BATCH_SIZE 16 OUTPUT_DIR /data04/junyi/results/mvit/{mode}_{num}_{aug_num}_ DATA_MODE {mode} DATA_AUG_NUM {aug_num} DATA_NUM {num}',shell=True)
#             # if num == 1 :
#             #     break
if __name__ == "__main__":
    aug_num_lst =[4,10,]
    GPU_lst = [3 ,3,3]
    with Pool(1) as p:
        p.starmap(run,zip(aug_num_lst,GPU_lst))
        