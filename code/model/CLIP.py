import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import pearsonr

from .nb_module import *
from . import vision_transformer as vits
from . import config as CFG
from .modules import ImageEncoder, TextEncoder, ProjectionHead

class STain(nn.Module):
    """
    STain: Predicting Gene Expression from H&E images, with ST Foundation model guids.
    
    Dataset:
    batch[0]: center image [B, 3, 224, 224]
    batch[1]: gene embedding [B, E] 
    batch[2]: logged gene expression [B, G]
    batch[3]: neighbor images [B, K, 3, 224, 224]
    batch[4]: spot diagram [B]
    """
    
    def __init__(self, temperature=CFG.temperature,
                 image_embedding_dim=CFG.image_embedding,
                 text_embedding_dim=CFG.text_embedding,
                 CLIP = True,
                 nbrs = True,
                 FiLM = True):
        super().__init__()
        
        # Image Encoder - Image feature extraction，Input: [B, 3, 224, 224], Output: [B, 384]
        def to_bool(value):
            if isinstance(value, str):
                return value.lower() == 'true'
            return bool(value)
        self.CLIP = to_bool(CLIP)
        self.nbrs = to_bool(nbrs)
        self.FiLM = to_bool(FiLM)
        self.image_encoder = ImageEncoder(pretrained=True).to(CFG.float)
        
        # Projection Head - Project image and text features to embedding dimension
        self.image_projection = ProjectionHead(embedding_dim=image_embedding_dim)  # output: [B, 512]
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim).to(CFG.float)  # output: [B, 512]
        
        # 2. Neighbor encode and Fusion
        # 1D Convolution - Fusion of neighbor images and centor image
        # input: [B, 384, K], output: [B, 512, K]
        self.conv = nn.Conv1d(
            in_channels=384,     
            out_channels=512,   
            kernel_size=9        
        )
        self.activation = nn.ReLU()
        

        # 3. FiLM
        self.diam_encoder = nn.Embedding(num_embeddings=5, embedding_dim=128)
        
        # FiLM layer- Create conditioned parametor 
        # input: [B, 128], output: [B, 1024] -> chunks are 2 [B, 512]
        self.film_shared = nn.Linear(128, 2 * 512)
        
        self.ln_img = nn.LayerNorm(512)  # Image Embedding Norm
        self.ln_ctx = nn.LayerNorm(512)  # Contextue Embedding Norm
        
        # 4. Gene Decoder
        # Image Decoder - Decode image embedding to gene expression prediction
        # input: [B, 1024], output: [B, CFG.hvg_num]
        self.image_decoder = nn.Sequential(
            nn.Linear(512 , 1024),  
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048), 
            nn.GELU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(2048, CFG.hvg_num),  
            nn.ReLU(),
        ).to(CFG.float)
        
        # 5. Loss weights
        self.temperature = float(temperature) 
        self.alpha = 1  
        self.beta = 5   

    def forward(self, batch, epoch=None, step_2=2, use_gene_expression=True, 
                weighted_loss=False, use_scgpt=True):
        """
        Feedforward broadcasting
        
        Args:
            batch: NumpyDataset
                [0]: Center patch i [B, 3, 224, 224]
                [1]: Gene feature [B, E] 
                [2]: Logged genen counts [B, G]
                [3]: Neighbour patches [B, K, 3, 224, 224] 
                [4]: Diagram [B]
            epoch: epoch number
            use_gene_expression: use gene expression if in training step, else no gene expression need 
            weighted_loss: if use weighted loss
            
        Returns:
            train mode: total_loss, decode_loss, contrastive_loss, pcc_list
            test mode: decoded_gene_expression [B, G]
        """
        print(f'self clip nbrs film:{self.CLIP}{self.nbrs}{self.FiLM}')

        # Image i Feature Extraction
        image_embeddings = self.image_encoder(batch[0].to(CFG.float)) 

        # neighbor encode and fusion
        split_tensors = batch[3]

        if self.nbrs==True:  
            with torch.no_grad():
                B, K = split_tensors.shape[0], split_tensors.shape[1]

                nbr_patches = split_tensors.reshape(B * K, *split_tensors.shape[2:])  

                # get neighbor patches encoded
                patch_nbrs = self.image_encoder(nbr_patches.to(CFG.float))  
                patch_nbrs = patch_nbrs.reshape(B, K, -1)  

            # neighbor feature fusion
            x = patch_nbrs.permute(0, 2, 1)  
            x = self.activation(self.conv(x))  
            c = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  
            c = self.ln_ctx(c)  

            image_with_nbrs = torch.cat([image_embeddings, c], dim=-1)  # [B, 384 + 512]
        else:
            image_with_nbrs = image_embeddings 
            c = torch.zeros(image_embeddings.size(0), 512).to(image_embeddings.device)
            image_with_nbrs = torch.cat([image_embeddings, c], dim=-1) 

        # Feature projection
        image_embeddings_proj = self.image_projection(image_with_nbrs)  
        image_embeddings_proj = self.ln_img(image_embeddings_proj) 

        # Diagram Conditional Module (FiLM)
        if self.FiLM==True:
            diam_idx = batch[4].long().to(image_embeddings_proj.device)
            z = self.diam_encoder(diam_idx) 

            # FiLM Layer
            z1, z2 = self.film_shared(z).chunk(2, dim=-1) 

            image_embeddings_film = (1 + z1) * image_embeddings_proj + z2 

            final_features = image_embeddings_film

            image_for_contrastive = image_embeddings_film
        else:
            final_features = image_embeddings_proj  
            image_for_contrastive = image_embeddings_proj

        # train or test
        if use_gene_expression:
            if not use_scgpt:
                text_embeddings = self.text_projection(batch[2].to(CFG.float))
            else:
                text_embeddings = self.text_projection(batch[1].to(CFG.float))
            decoded_image_embeddings = self.image_decoder(final_features) 
            if self.CLIP==True:
                logits = (text_embeddings @ image_for_contrastive.T) / self.temperature  
                images_similarity = F.normalize(image_for_contrastive @ image_for_contrastive.T, dim=-1)  # [B, B]
                texts_similarity = F.normalize(text_embeddings @ text_embeddings.T, dim=-1) 

                targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)  # [B, B]

                texts_loss = F.cross_entropy(logits, targets, reduction='none') 
                images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')  
                contrastive_loss = (images_loss + texts_loss) / 2.0 
                alpha = 1
            else:
                contrastive_loss=torch.zeros(image_embeddings.size(0)).to(image_embeddings.device)
                alpha = 0

            # Reconstruction Loss
            decode_loss = F.mse_loss(decoded_image_embeddings, batch[2].to(CFG.float))

            # Weighted loss
            if weighted_loss:
                weights = update_weights(decode_loss) 
                decode_loss = decode_loss * weights

            # Loss
            beta = 5

            total_loss = alpha * contrastive_loss.mean() + beta * decode_loss.mean()

            # PCC for eval
            pcc = [
                pearsonr(
                    decoded_image_embeddings[:, i].detach().cpu().numpy(),
                    batch[2][:, i].detach().cpu().numpy()
                ).statistic
                for i in range(decoded_image_embeddings.shape[1])
            ]
            print(f'total loss: {total_loss:.4f}', f'\t recon loss: {decode_loss.item():.4f}', f'\t con loss: {contrastive_loss.mean():.4f}')
            return total_loss, beta * decode_loss.mean(), alpha * contrastive_loss.mean(), pcc

        else:
            # test mode
            decoded_image_embeddings = self.image_decoder(final_features)
            return decoded_image_embeddings

class STain_test_dataloader(nn.Module):
    """
    STain: Predicting Gene Expression from H&E images, with ST Foundation model guids.
    
    Dataset:
    batch[0]: center image [B, 3, 224, 224]
    batch[1]: gene embedding [B, E] 
    batch[2]: logged gene expression [B, G]
    batch[3]: neighbor images [B, K, 3, 224, 224]
    batch[4]: spot diagram [B]
    """
    
    def __init__(self, temperature=CFG.temperature,
                 image_embedding_dim=CFG.image_embedding,
                 text_embedding_dim=CFG.text_embedding,
                 CLIP = True,
                 nbrs = True,
                 FiLM = True):
        super().__init__()
        
        def to_bool(value):
            if isinstance(value, str):
                return value.lower() == 'true'
            return bool(value)
        self.CLIP = to_bool(CLIP)
        self.nbrs = to_bool(nbrs)
        self.FiLM = to_bool(FiLM)
        self.image_encoder = ImageEncoder(pretrained=True).to(CFG.float)
        
        # Projection Head - Project image and text features to embedding dimension
        self.image_projection = ProjectionHead(embedding_dim=image_embedding_dim)  # output: [B, 512]
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim).to(CFG.float)  # output: [B, 512]
        
        self.conv = nn.Conv1d(
            in_channels=384,     
            out_channels=512,    
            kernel_size=9        
        )
        self.activation = nn.ReLU()

        self.diam_encoder = nn.Embedding(num_embeddings=5, embedding_dim=128)
        
        self.film_shared = nn.Linear(128, 2 * 512)
        
        # 层归一化
        self.ln_img = nn.LayerNorm(512)  
        self.ln_ctx = nn.LayerNorm(512)  
        
        self.image_decoder = nn.Sequential(
            nn.Linear(512 , 1024),  
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048), 
            nn.GELU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(2048, CFG.hvg_num),  
            nn.ReLU(),
        ).to(CFG.float)

        self.temperature = float(temperature)  
        self.alpha = 1  
        self.beta = 5  

    def forward(self, batch, epoch=None, step_2=2, use_gene_expression=True, 
                weighted_loss=False, use_scgpt=True):
        print('Forward Pass')
        total_loss = 1.0
        contrastive_loss= 1.0
        decode_loss = 1.0
        pcc = 1.0
        return total_loss, beta * decode_loss.mean(), alpha * contrastive_loss.mean(), pcc

if __name__ == '__main__':
    batch_size = 8
    num_genes = CFG.hvg_num
    num_neighbors = 8  
    embedding_dim = 384  
    

    test_batch = [
        torch.randn(batch_size, 3, 224, 224),      
        torch.randn(batch_size, embedding_dim),   
        torch.randn(batch_size, num_genes),        
        torch.randn(batch_size, num_neighbors, 3, 224, 224),  
        torch.randint(0, 5, (batch_size,))        
    ]
    
    model = STain()
    
    print("TESTING...")
    total_loss, decode_loss, contrastive_loss, pcc_list = model(test_batch, use_gene_expression=True)
    print(f"total_loss: {total_loss:.4f}")
    print(f"reconstruction_loss: {decode_loss:.4f}") 
    print(f"contrastive_loss: {contrastive_loss:.4f}")
    print(f"Pearson_Correlation: {len(pcc_list)}")
    print("\ntesting mode...")
    predictions = model(test_batch, use_gene_expression=False)
    print(f"Gene Prediction shape: {predictions.shape}")
    
    print("Done!")