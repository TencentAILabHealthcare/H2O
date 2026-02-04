import torch
from pathlib import Path

debug = False
batch_size = 32 #32
num_workers = 0
lr = 1e-4
weight_decay = 5e-5
patience = 2
factor = 0.5
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_ckpt = False

#scgpt config
d_model = 512
mvc_decoder_style = "inner product" #1. "inner product" or 2. "concat query" or 3. "sum query".
explicit_zero_prob = False
use_batch_labels = False


model_name = 'vit_base_patch16_224'
vit_pth = 'jx_vit_base_p16_224-80ecf9dd.pth'
image_ckpt = './checkpoint/TCGA_all_32cancer_vit_small/checkpoint.pth'

scale = 'cell'
seq_length = 1500 # if scale is gene

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
#projection_dim = 384
image_embedding = 384 + 512
text_embedding = 512 #5000#512


n_top = 1000
projection_dim = 512
dropout = 0.1 #in running
repeat_times = 5
float = torch.float32
iscut = False

hvg_num = 5033#785#12482
hvg_num_ccrcc = 1860
weighted_loss = False
# nohup python main_APE.py > a100_7500_masked.log 2>&1 &