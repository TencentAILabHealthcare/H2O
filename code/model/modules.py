import os
import torch
from torch import nn
import pandas as pd
import timm
from . import config as CFG
from . import vision_transformer as vits
from . import vision_transformer_patch_token as vits_patch

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from argparse import Namespace
import sys
import copy
import json
import os
from pathlib import Path
import warnings

sys.path.insert(0, "../")
#import scGPT.scgpt as scg
#from scGPT.scgpt.tasks import GeneEmbedding
#from scGPT.scgpt.tokenizer.gene_tokenizer import GeneVocab
#from scGPT.scgpt.model import TransformerModel
#from scGPT.scgpt.preprocess import Preprocessor
#from scGPT.scgpt.utils import set_seed 


class TextEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, ckpt_path, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model,self.model_config =  load_model_frommmf(ckpt_path)
        self.token_emb =self.model.token_emb(require_grad=False)
        self.pos_emb = self.model.pos_emb.eval(require_grad=False)
        self.encoder =self.model.encoder.eval(require_grad=False)
        self.fc1=nn.Linear(512,1)

    def forward(self, x):
        #value_labels = x > 0
        #x, x_padding = gatherData(x, value_labels,self.model_config['pad_token_id'])
        #self.model.eval()
        self.model.to_final = None
        encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(x,x,self.model_config)
        
        out=self.model.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,)  
        
        out = out[:,:,:].contiguous()
        out=self.fc1(out)
        return out

class TextEncoder_freeze(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, ckpt_path, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(19266, 12288),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(12288, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 384),
        )

    def forward(self, x):
        out = self.linear_layers(x)
        return out

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

class TextEncoder_scgpt_freeze(nn.Module):
    """
    Encode expressions to a fixed size vector
    """

    def __init__(
        self, ckpt_path=Path("scGPT/scgpt/save/scGPT_bc"), pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model_dir = Path("../save/scGPT_bc")
        self.preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=3,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=False,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=19264,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="cell_ranger",
            #binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
            )
        self.embed_data=scg.tasks.embed_data()


    def forward(self, x):
        set_seed(42)
        cell_type_key = None
        gene_col = "index"
        adata = self.preprocessor(x, batch_key="batch")
        embed_adata = scg.tasks.embed_data(
            adata,
            self.model_dir,
            gene_col=gene_col,
            obs_to_save=cell_type_key,  # optional arg, only for saving metainfo
            batch_size=CFG.batch_size,
            return_new_adata=True,
                )
        out = embed_adata
        return out

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

vit_base_patch16_224_default_cfg = {
    'url': CFG.vit_pth,
    'num_classes': 1000,
    'input_size': (3, 224, 224),
    'pool_size': None,
    'crop_pct': 0.9,
    'interpolation': 'bicubic',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'first_conv': 'patch_embed.proj',
    'classifier': 'head',
}

default_cfg = {}
default_cfg['vit_base_patch16_224'] = vit_base_patch16_224_default_cfg

class ImageEncoder_raw(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, #@global_pool="avg",
            default_cfg=default_cfg[model_name] 
        )
        for p in self.model.parameters():
            p.requires_grad = trainable
        load_pretrained_weights(self.model, CFG.image_ckpt, 'teacher', model_name, 16)
        self.model.eval()

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model =  vits.__dict__['vit_small'](patch_size=8, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = trainable
        load_pretrained_weights(self.model, CFG.image_ckpt, 'teacher', model_name, 16)

    def forward(self, x):
        return self.model(x)

class ImageEncoder_patchtoken(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=False, trainable=CFG.trainable
    ):
        super().__init__()
        self.model =  vits_patch.__dict__['vit_small'](patch_size=8, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = trainable
        load_pretrained_weights(self.model, CFG.image_ckpt, 'teacher', model_name, 16)

    def forward(self, x):
        return self.model.get_patch_tokens(x)

class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb, gene_embs
    ):
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)

#
def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var



def exists(val):
    return val is not None

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        # print("x_mask",x_mask_idx.shape,x_mask_idx)
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        # print('bin_num_idx', bin_num_idx.shape)
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        # print('token_emb', token_emb.shape)
        x = torch.matmul(weight, token_emb) #[B, N, D]
    
        # print("x_emb",x.shape,x)
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x

class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class scEncoder(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,  # num of tokens
            max_seq_len,  # max length of sequence
            embed_dim,  # encoder dim of tokens
            decoder_embed_dim,
            tie_embed=False,
            bin_alpha = 1.0,
            bin_num = 10,
            pad_token_id = None,
            mask_token_id = None,
    ):
        super(scEncoder, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=self.pad_token_id, mask_token_id=self.mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim)  #RandomPositionalEmbedding(embed_dim, max_seq_len)

        # ## DEBUG
        self.encoder = None

        ##### decoder
        self.decoder = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

    def forward(self, x, padding_label, encoder_position_gene_ids, encoder_labels, decoder_data,
                mask_gene_name, mask_labels, decoder_position_gene_ids, decoder_data_padding_labels,
                output_attentions=False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight = 0)
        if output_attentions:
            x.requires_grad_()  # used for attn_map output

        position_emb = self.pos_emb(encoder_position_gene_ids)
        x += position_emb
        x = self.encoder(x, padding_mask=padding_label)



        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)
        if mask_gene_name:
            # todo
            # mask gene_name
            print('mask_gene_name not done')
            exit(0)
        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        decoder_data += position_emb

        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        # print("x0",x.shape) 
        x = self.norm(x)
        # print("x1",x.shape) 
        if exists(self.to_final):
            x = self.to_final(x)
            return x.squeeze(2) 
        else:
            return x
        return x

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

