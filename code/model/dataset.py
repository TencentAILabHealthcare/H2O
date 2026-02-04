# numpydataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms as pth_transforms
import h5py
from PIL import Image
from typing import List
import numpy as np
import os
import time



class FixedNumpyDataset(Dataset):
    """
    Dataset for loading imgs and expresssions:

    """
    
    def __init__(self, 
                 ids: List[int], 
                 h5_path: str, 
                 mode: str = 'train',
                 preload_to_memory: bool = False):
        """
            ids: samplle ids
            h5_path: HDF5 file path
            mode: 'train' or 'test'. Use image augmentation if 'train'
            preload_to_memory: whether to load all the data into memory
        """
        self.ids = ids
        self.h5_path = h5_path
        self.mode = mode
        self.preload = preload_to_memory
        
        print(f"[{mode.upper()}] Intializing FixedNumpyDataset:")
        print(f"  Number of samples: {len(ids)}")
        print(f"  Files: {os.path.basename(h5_path)}")
        print(f"  Preload: {preload_to_memory}")
        
        # 根据模式创建transform
        if self.mode == 'train':
            self.transform = pth_transforms.Compose([
                pth_transforms.Resize(256, interpolation=Image.BILINEAR),
                pth_transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                pth_transforms.RandomHorizontalFlip(p=0.5),
                pth_transforms.RandomVerticalFlip(p=0.5),
                pth_transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02
                ),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            print("  Use transform")
        else:
            self.transform = None
            print("  Ignore transform")
        
        if self.preload:
            self._preload_data()
        else:
            self._validate_file()
    
    def _validate_file(self):
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 does not exist: {self.h5_path}")
        
        file_size = os.path.getsize(self.h5_path) / 1024 / 1024 / 1024
        print(f"  File size: {file_size:.2f} GB")
        
        with h5py.File(self.h5_path, 'r') as f:
            required_keys = ['emb', 'counts', 'img', 'nbrs']
            missing_keys = [k for k in required_keys if k not in f]
            if missing_keys:
                raise KeyError(f"Key file absence: {missing_keys}")
    
    def _preload_data(self):
        print("  Preloading...")
        start_time = time.time()
        
        try:
            with h5py.File(self.h5_path, 'r') as f:

                self.emb_data = f['emb'][:]
                self.counts_data = f['counts'][:]
                self.img_data = f['img'][:]
                
                self.nbrs_data = f['nbrs'][:] if 'nbrs' in f else None
                self.diam_data = f['diam'][:] if 'diam' in f else None
            
            elapsed = time.time() - start_time
            print(f"  Preloading Finished: {elapsed:.2f} seconds")
            
            total_mb = (self.emb_data.nbytes + self.counts_data.nbytes + 
                       self.img_data.nbytes) / 1024 / 1024
            print(f"  Memory: {total_mb:.2f} MB")
            
        except Exception as e:
            print(f"  Fail: {e}")
            raise
    
    def _load_single_sample(self, iterid: int) -> dict:
        if self.preload:
            return {
                'emb': self.emb_data[iterid],
                'counts': self.counts_data[iterid],
                'img': self.img_data[iterid],
                'nbrs': self.nbrs_data[iterid] if self.nbrs_data is not None else [],
                'diam': self.diam_data[iterid] if self.diam_data is not None else -1
            }
        else:
            with h5py.File(self.h5_path, 'r') as f:
                return {
                    'emb': f['emb'][iterid],
                    'counts': f['counts'][iterid],
                    'img': f['img'][iterid],
                    'nbrs': f['nbrs'][iterid] if 'nbrs' in f else [],
                    'diam': f['diam'][iterid] if 'diam' in f else -1
                }
    
    def _load_neighbor_images(self, nbr_indices: List[int]) -> List[np.ndarray]:
        neighbor_images = []
        
        if len(nbr_indices) == 0:
            return neighbor_images
        
        if self.preload:
            for idx in nbr_indices:
                neighbor_images.append(self.img_data[idx])
        else:
            with h5py.File(self.h5_path, 'r') as f:
                for idx in nbr_indices:
                    neighbor_images.append(f['img'][idx])
        
        return neighbor_images
    
    def _process_image(self, img_np: np.ndarray) -> torch.Tensor:
        """处理图像：根据模式决定是否使用transform"""
        if self.transform is not None:
            img = Image.fromarray(img_np.astype('uint8'))
            return self.transform(img)
        else:
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
            
            img = Image.fromarray(img_np)
            
            basic_transform = pth_transforms.Compose([
                pth_transforms.Resize(224),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            return basic_transform(img)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        iterid = self.ids[idx]
        
        data = self._load_single_sample(iterid)
        
        x = torch.from_numpy(data['emb']).float()
        y = torch.from_numpy(data['counts']).float()
        
        img_center = self._process_image(data['img'])
        
        nbrs_indices = data['nbrs']
        nbrs_imgs = []
        
        if len(nbrs_indices) > 0:
            neighbor_np_imgs = self._load_neighbor_images(nbrs_indices)
            for nbr_img_np in neighbor_np_imgs:
                nbr_img = self._process_image(nbr_img_np)
                nbrs_imgs.append(nbr_img)
            
            if nbrs_imgs:
                nbrs_imgs = torch.stack(nbrs_imgs, dim=0)
            else:
                nbrs_imgs = torch.zeros(0, *img_center.shape)
        else:
            nbrs_imgs = torch.zeros(0, *img_center.shape)
        
        diam_raw = data['diam']
        diam_to_index = {'2': 0, '55': 1, '100': 2, '150': 3, '-1': 4}
        
        try:
            if isinstance(diam_raw, (np.ndarray, list)) and len(diam_raw) > 0:
                diam_value = str(int(diam_raw[0]))
            else:
                diam_value = str(int(diam_raw))
            
            diam_idx = torch.tensor(diam_to_index.get(diam_value, 4), dtype=torch.long)
        except Exception:
            diam_idx = torch.tensor(4, dtype=torch.long)
        
        return [img_center, x, y, nbrs_imgs, diam_idx,iterid]
class NumpyDataset(Dataset):
    def __init__(self, ids, h5_path, transform):
        self.ids = ids
        self.h5_path = h5_path
        self.transforms = transform
        self._h5_ref = None  

    def __len__(self):
        return len(self.ids)

    @property
    def h5(self):
        if self._h5_ref is None:
            self._h5_ref = h5py.File(self.h5_path, 'r')
        return self._h5_ref

    def __getitem__(self, idx):
        #print(self.ids[idx])
        iterid = self.ids[idx]

        x_np = self.h5['emb'][iterid]         
        y_np = self.h5['counts'][iterid]      
        x = torch.from_numpy(x_np).float()     
        y = torch.from_numpy(y_np).float()     

        # ---- center patch ----
        img_np = self.h5['img'][iterid].astype('uint8')  
        img = Image.fromarray(img_np)
        img = self.transforms(img)                     

        # ---- diagram -> index ----
        if 'diam' in self.h5:
            diam_raw = self.h5['diam'][iterid]
            diam_to_index = {'2': 0, '55': 1, '100': 2, '150': 3, '-1': 4}
            
            try:
                diam_idx = torch.tensor(diam_to_index[str(int(diam_raw))], dtype=torch.long)
            except Exception:
                diam_idx = torch.tensor([diam_to_index[str(int(d))] for d in diam_raw], dtype=torch.long)
        else:
            diam_idx = torch.tensor(4, dtype=torch.long)

        nbrs_indices = self.h5['nbrs'][iterid]    
        nbrs_imgs = []
        for i in nbrs_indices:
            nbr_img_np = self.h5['img'][i].astype('uint8')
            nbr_img = self.transforms(Image.fromarray(nbr_img_np)) 
            nbrs_imgs.append(nbr_img)
        nbrs_imgs = torch.stack(nbrs_imgs, dim=0)

        return [img, x, y, nbrs_imgs, diam_idx] 

    def __del__(self):
        try:
            if self._h5_ref is not None:
                self._h5_ref.close()
        except Exception:
            pass
class CLIPDataset_sc(torch.utils.data.Dataset):
    def __init__(self,image_filenames, captions,rawcaptions, captionindex, tissues, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = captions
        self.captions_raw = rawcaptions
        self.transforms = transforms
        self.tissues = tissues 
        self.captionindex = captionindex

    def __getitem__(self, idx):
        item = {}
        captionindex = torch.tensor(self.captionindex[idx]).clone().detach()
        item['captionindex'] = captionindex
        image_path = self.image_filenames[idx]
        #item['image_path']= torch.tensor(image_path).clone().detach()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        image = Image.fromarray(image)
        image = self.transforms(image)#['image']
        item['image'] = torch.tensor(image).float() #.permute(2, 0, 1)
        #totalcount = self.captions.iloc[idx,:].sum()
        totalcount = self.captions[idx].sum()
        totalcount = np.log10(totalcount)
        #tmpdata = self.captions.iloc[idx,:].tolist()
        tmpdata = self.captions[idx].tolist()
        #tmpdataraw = self.captions_raw.iloc[idx,:].tolist()
        tmpdataraw = self.captions_raw.iloc[idx,:].tolist()
        pretrain_gene_x = torch.tensor(tmpdata+[totalcount,totalcount])
        pretrain_gene_x_raw = torch.tensor(tmpdataraw+[totalcount,totalcount])
        item['caption'] = pretrain_gene_x.clone().detach()
        item['caption_raw'] = pretrain_gene_x_raw.clone().detach()
        return item


    def __len__(self):
        return len(self.captions)
        
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )