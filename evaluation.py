import numpy as np
import nibabel as nib
import os
from itertools import product
import pathlib
import pandas as pd
try:
  from typing import Literal
  ModelName = Literal["MViTv2-B", "MViTv2-S", "MViTv2-L", "MViTv2-T", ]
except ImportError:
  ModelName = str
import torch
import torchvision.transforms.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score,confusion_matrix
# os.environ['LD_LIBRARY_PATH'] = ''
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


from matplotlib import pyplot as plt
np.load('/data04/301_postprocess/301_embed/tractoembedding/sub-14117/tractoembedding/da-full/fiber_indices.npy')


def load_model_and_preproc(model_id,data_id,model="MViTv2-mri", use_cuda:bool=torch.cuda.is_available(), use_half:bool=True):
    """
    Load an MViT2 model and get the image transformer
    :param model: the name of the model. Should be  one of "MViTv2-B", "MViTv2-S", "MViTv2-L", "MViTv2-T". Defaults to "MViTv2-B".
    :param use_cuda: whether to use cuda. Defaults to True if cuda is available.
    :param use_half: whether to use half-precision. Defaults to True, but will not be used if cuda is not used.
    :returns a tuple of (the model, image transformer)
    """
    model = model.replace('-','_')
    use_half &= use_cuda

    import urllib.request
    from mvit.config.defaults import get_cfg
    from mvit.models import build_model
    from mvit.datasets import Tractoembedding
    from mvit.datasets import loader
    mode = 'fusion'
    ckpt_path = f'/data06/junyi/results/mvit_direct_voting_weighted_new/fusion_3_010_{model_id}/checkpoints/checkpoint_epoch_00130.pyth'
    # ckpt_path = f"/data04/junyi/results/mvit_balanced_fold_new_norm_direct/{mode}_3_030_0123/checkpoints/checkpoint_epoch_00160.pyth"
    # if not os.path.exists(ckpt_path):
    #     urllib.request.urlretrieve(f"https://dl.fbaipublicfiles.com/mvit/mvitv2_models/{ckpt_path}", ckpt_path)
    cfg = get_cfg()
    
    # cfg.MVIT.CLS_EMBED_ON = True
    cfg.DATA.PATH_TO_DATA_DIR = 'fold.csv'
    # cfg.merge_from_file(f"./configs/test/{model.replace('ViT','VIT')}_test.yaml")
    cfg.merge_from_file('configs/MVITv2_mri.yaml')
    cfg.DATA.MODE = 'fusion'
    cfg.MVIT.CLS_EMBED_ON = True
    cfg.DATA_AUG_NUM = 10
    cfg.DATA_NUM = 3
    cfg.NUM_GPUS = 1
    cfg.DATA_MODE = mode
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    
    model.eval()
    # if use_half:
    #     model = model.half()
    print(model)
    cfg.TRAIN.BATCH_SIZE = 2
    return model,loader.construct_loader(cfg, "val",data_id)
# input_images,labels = next(iter(datagen))

for model_id in range(5):
    preds_list = []
    labels_list = []
    attn_lists = []
    folds = pd.DataFrame()
    for data_id in range(5):
        model, datagen = load_model_and_preproc(model_id,data_id)
        fold = pd.read_csv('fold.csv')
        fold = fold[fold['fold']==data_id]
        folds = pd.concat([folds,fold])
        for input_images,labels in datagen:
            allimages = []
            for images4model in input_images[0]:
                imagecuda = [image.cuda() for image in images4model]
                allimages.append(imagecuda)
            attn_list = []
            with torch.no_grad():
                model.eval()
                preds = model(allimages).cpu()
                # preds = torch.argmax(preds,dim=1)
                preds_list.append(preds)
                labels_list.append(labels)
            # for mi,pi in product(range(3),range(3)):
            #     attns = model.model_list[mi].blocks[2*pi].attn.attn_lst[:,:,0,1:]
            #     print(attns.shape)
            # attn_list.append(attns.cpu())
            # preds = (preds>0.5).int().squeeze()
        torch.cuda.empty_cache()
        attn_lists.append(attn_list)
    preds = torch.cat(preds_list)
    print(preds)

    labels = torch.cat(labels_list)
    # attns_list = [attn for attn_list in attn_lists for attn in attn_list]
    labels = 1-labels
    preds = preds[:,0]
    folds['preds'] = preds
    folds['labels'] = labels
    folds.to_csv(f'fold_{model_id}.csv',index=False)
# fpr,tpr,threshold = roc_curve(labels,preds)
# plt.plot(fpr,tpr)
# plt.plot([0,1],[0,1],'--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.savefig('roc.png')
# # labels = labels.numpy().astype(bool)
# # print('%d,%d,%d',precision_score(labels,preds>0.5),recall_score(labels,preds>0.5),f1_score(labels,preds>0.5))
# for threshold in np.linspace(0,1,40):
#     print('Threshold:',threshold)

#     tn, fp, fn, tp = confusion_matrix(labels,preds>threshold).ravel()
#     print(tn, fp, fn, tp)
#     # print('Accuracy: {:.3f}, Specificity: {:.3f}, Sensitivity: {:.3f}'.format((mat[0,0]+mat[1,1])/mat.sum(),mat[0,0]/mat[0].sum(),mat[1,1]/mat[1].sum()))
#     print('Accuracy: {:.3f}'.format((tn+tp)/(tn+fp+fn+tp)))
#     print('Specificity: {:.3f}, Sensitivity: {:.3f}'.format(tn/(tn+fp),tp/(tp+fn)))
#     print('AUC: {:.3f}'.format(roc_auc_score(labels,preds)))
# mixed = [((labels==0)&(preds==labels)).float().sum(),
#          ((labels==0)&(preds!=labels)).float().sum(),
#          ((labels==1)&(preds==labels)).float().sum(),
#          ((labels==1)&(preds!=labels)).float().sum()]
# print(mixed)
# prec = mixed[0]/(mixed[0]+mixed[1])
# recall = mixed[0]/(mixed[0]+mixed[3])
# f1 = 2*prec*recall/(prec+recall)
# print(f'Precision: {prec:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')

# precison_list = ['sz80','sz160','sz320',]
# precison_list.reverse()
# mode_list = ['commisural','left','right',]
# m = labels == preds
# print(len(m))

# for mi,pi in product(range(3),range(3)):
#     attns = model.model_list[mi].blocks[2*pi].attn.attn_lst[m,:,0,1:].mean(0,keepdim=True)
#     B,nH,nP = attns.shape
#     fig = plt.figure(figsize=(8,5))
#     mask_total = torch.zeros((1,20,20))
#     mean,std = attns.mean().cpu(),attns.std().cpu()
#     for head in range(nH):
#         attn = attns[:,head,].reshape(-1,20,20).cpu()
#         mask_total+=attn
#         mask = F.resize(attn, input_images[0][0][0].shape[-2:], F.InterpolationMode.BILINEAR)
#         # mask = mask[m,:,:].mean(0)
#         # mask_total+=mask
        
#         np.save(f'./attns/precision_{precison_list[pi]}_mode_{mode_list[mi]}_head_{head}.npy',mask)
#         ax = plt.subplot(1,nH,head+1,frameon=False)
#         # mean,std = mask.mean(),mask.std()
#         mask = (mask-mean)/std
#         # one side 95% confidence interval
#         mask[mask<3.29] = 0
#         ax.imshow(mask[0],cmap='jet',)
#         ax.axis('off')
#         ax.set_title(f'Head: {head}',fontsize=8)
        
#         # np.save(f'./attns/precision_{precison_list[pi]}_mode_{mode_list[mi]}_head_{head}.npy',mask)
#     mask_total/=nH
#     mask_total = F.resize(mask_total, input_images[0][0][0].shape[-2:], F.InterpolationMode.BILINEAR)
#     np.save(f'./attns/precision_{precison_list[pi]}_mode_{mode_list[mi]}_total.npy',mask_total.numpy())
#     fig.suptitle(f'Precision: {precison_list[pi]}, Mode: {mode_list[mi]}',fontsize=10,fontweight='bold',y=0.95)
#     fig.tight_layout()
#     pathlib.Path('./attns').mkdir(exist_ok=True)
#     fig.savefig(f'./attns/precision_{precison_list[pi]}_mode_{mode_list[mi]}.png',dpi=300)