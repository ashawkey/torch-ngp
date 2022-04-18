import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import clip

class CLIPLoss:
    def __init__(self, device, name='ViT-B/16'):
        self.device = device
        self.name = name
        self.clip_model, self.transform_PIL = clip.load(self.name, device=self.device, jit=False)

        # disable training
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # image augmentation
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # placeholder
        self.text_zs = None
        self.image_zs = None
    
    def normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True)
    
    # image-text (e.g., dreamfields)
    def prepare_text(self, texts):
        # texts: list of strings.
        texts = clip.tokenize(texts).to(self.device)
        self.text_zs = self.normalize(self.clip_model.encode_text(texts))
        print(f'[INFO] prepared CLIP text feature: {self.text_zs.shape}')
    
    def __call__(self, images, mode='text'):

        images = self.transform(images)
        image_zs = self.normalize(self.clip_model.encode_image(images))

        if mode == 'text':
            # if more than one string, randomly choose one.
            if self.text_zs.shape[0] > 1:
                idx = random.randint(0, self.text_zs.shape[0] - 1)
                text_zs = self.text_zs[[idx]]
            else:
                text_zs = self.text_zs
            # broadcast text_zs to all image_zs
            loss = - (image_zs * text_zs).sum(-1).mean()
        else:
            raise NotImplementedError

        return loss
    
    # image-image (e.g., diet-nerf)
    def prepare_image(self, dataset):
        # images: a nerf dataset (we need both poses and images!)
        pass