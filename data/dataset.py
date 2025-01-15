import os
import csv
import glob
import time
import numpy as np
import pickle
import torch
import einops
import clip
from torchvision import datasets, transforms
from PIL import Image

class MMJDM(torch.utils.data.Dataset):
    def __init__(self, args, is_train=True, transform=None):
        self.data_paths = args.data_paths
        self.dataset = args.dataset
        self.transform = transform
        self.action = args.action
        self.is_train = is_train
        self.length = args.length 

        self.rgb_data_path = self.data_paths['train_rgb_data']
        self.flow_data_path = self.data_paths['train_flow_data']
        self.skeleton_data_path = self.data_paths['train_skeleton_data']
        self.text_data_path = self.data_paths['train_text_data']
        self.skeleton_data = np.load(os.path.join(self.skeleton_data_path, 'skeleton_data_alignment.npy'))
        self.CLIP, self.CLIPImgPreprocess = clip.load("ViT-B/16", device="cuda")

        self.pretrain_feats = args.pretrain_feats
        self.feat_label_path = args.feat_label_path
        self.feat_clip_path = args.feat_clip_path
        self.feat_root_path = args.feat_root_path
        self.text_dict = self.read_pickle(self.text_data_path) 
        
        self.miss_modal = args.miss_modal
        self.miss_modal_rate = 0
        self.miss_label_rate = 0

        if is_train:
            self.split_path = args.train_split_path
        else:
            self.split_path = args.test_split_path
        self.split = self.read_pickle(self.split_path)          

        if args.miss_modal_rate > 0:
            self.miss_modal_rate = args.miss_modal_rate
            self.miss_modal_split = np.random.choice(self.split, int(len(self.split) * self.miss_modal_rate), replace=False)

        if args.miss_label_rate > 0:
            self.miss_label_rate = args.miss_label_rate
            self.miss_lable_split = np.random.choice(self.split, int(len(self.split) * self.miss_label_rate), replace=False)

        if self.dataset == 'Myositis' or self.dataset == 'MyositisClassify':
            self.label_path = args.label_path
        elif self.dataset == 'MyositisTotal':
            self.label_path = args.label_path_tot
            if is_train:
                self.split_path = args.train_split_path_tot
            else:
                self.split_path = args.test_split_path_tot
        self.label_dict = self.read_pickle(self.label_path)     


    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data
    
    def load_video(self, index):
        image_list = sorted(
            (glob.glob(os.path.join(self.rgb_data_path, f'{index:04d}*.jpg')))
        )
        video = [Image.open(image_list[i]) for i in range(self.length)]

        return video
        # return self.transform(video)

    def load_image_text(self, index, rgb):
        text = self.text_dict.get(index)['text']
        text = clip.tokenize(text, truncate=True).cuda() # context length is 77, so truncate

        with torch.no_grad():
            frame_list = [self.CLIPImgPreprocess(frame).unsqueeze(0).cuda() for frame in rgb]
            CLIPvideo = self.CLIP.encode_image(torch.cat(frame_list, dim=0)).float()
            text = self.CLIP.encode_text(text)

        return CLIPvideo, text
    
    def load_pretrain_feats(self, index):
        rgb_feats = np.load(os.path.join(self.feat_root_path, f'JDM{self.action}_rgb_VST.npy'), allow_pickle=True).item()
        flow_feats = np.load(os.path.join(self.feat_root_path, f'JDM{self.action}_flow_I3D.npy'), allow_pickle=True).item()
        with open(self.feat_label_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if int(row[1]) == index:
                    sample_name = row[0]
                    break
        rgb_feat = rgb_feats[sample_name]
        flow_feat = flow_feats[sample_name]
        return rgb_feat, flow_feat
    
    def load_clip_feats(self, index):
        CLIP_feats = np.load(os.path.join(self.feat_root_path, self.feat_clip_path), allow_pickle=True).item()
        return CLIP_feats[index]['video'], CLIP_feats[index]['text']
    
    def process_missing_modal(self, rgb, flow, skeleton, CLIPvideo, text):
        if 'V' in self.miss_modal:
            if type(rgb) == np.ndarray:
                rgb = np.zeros_like(rgb)
            else:
                rgb = torch.zeros_like(rgb)
        if 'F' in self.miss_modal or self.miss_modal_rate > 0:
            if type(flow) == np.ndarray:
                flow = np.zeros_like(flow)
            else:
                flow = torch.zeros_like(flow)
        if 'S' in self.miss_modal or self.miss_modal_rate > 0:
            if type(skeleton) == np.ndarray:
                skeleton = np.zeros_like(skeleton)
            else:
                skeleton = torch.zeros_like(skeleton)
        if 'T' in self.miss_modal or self.miss_modal_rate > 0:
            if type(CLIPvideo) == np.ndarray:
                CLIPvideo = np.zeros_like(CLIPvideo)
            else:
                CLIPvideo = torch.zeros_like(CLIPvideo)
            if type(text) == np.ndarray:
                text = np.zeros_like(text)
            else:                   
                text = torch.zeros_like(text)
        return rgb, flow, skeleton, CLIPvideo, text
    
    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        index = self.split[idx]     

        if self.dataset == 'Myositis':
            score = int(self.label_dict.get(index).get('score'))

        if self.pretrain_feats:   
            rgb_change, flow = self.load_pretrain_feats(index)
            rgb = self.load_video(index)
            CLIPvideo, text = self.load_clip_feats(index)
        else:
            rgb = self.load_video(index)
            flow = torch.tensor(np.load(os.path.join(self.flow_data_path, f'{index:04d}_op.npz'))['arr_0'])
            CLIPvideo, text = self.load_image_text(index, rgb)

            transformed_frames = [self.transform(frame) for frame in rgb]
            rgb = torch.stack(transformed_frames)
            rgb_change = einops.rearrange(rgb, 't c h w -> c t h w')    
            flow = einops.rearrange(flow, 't h w c -> c t h w')    

            flow = torch.nn.functional.interpolate(flow, size=(224, 224), mode='bilinear', align_corners=False)

        skeleton = torch.tensor(self.skeleton_data[index])

        if self.is_train and self.miss_label_rate > 0 and index in self.miss_lable_split:
            score = -1  

        if self.miss_modal or (self.miss_modal_rate > 0 and index in self.miss_modal_split):
            rgb_change, flow, skeleton, CLIPvideo, text = self.process_missing_modal(rgb_change, flow, skeleton, CLIPvideo, text)
    
        return rgb_change, flow, skeleton, CLIPvideo, text, score

def get_dataloader(args, is_train, shuffle=True, transform=None):
    CurDataset = MMJDM(args, is_train, transform=transform)

    if not is_train:
        return torch.utils.data.DataLoader(CurDataset, batch_size=args.batch_size, shuffle=shuffle, pin_memory=False)

    split = CurDataset.split
    label_dict = CurDataset.label_dict
    
    if CurDataset.dataset == 'Myositis':
        labels = [label_dict.get(split[i]).get('score') for i in range(len(split))]

    labels = np.array(labels).astype(int)
    counts = np.bincount(labels)
    weights = np.array([1.0 / counts[labels[i]] for i in range(len(labels))])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(labels))

    dataloader = torch.utils.data.DataLoader(CurDataset, batch_size=args.batch_size, pin_memory=False, sampler=sampler)

    return dataloader
