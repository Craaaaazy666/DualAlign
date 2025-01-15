import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.cross_attention import CrossAttention

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def eth( in_channels, num_classes, normal = False):
    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
    if normal:
        etf_vec = (etf_vec / torch.sum(etf_vec, dim=0, keepdim=True))
    return etf_vec, etf_rect

class DualAlign(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, n_head, n_query, dropout, no_etf):
        super(DualAlign, self).__init__()

        self.embedding_rgb = nn.Linear(768, in_dim)
        self.embedding_CLIPvideo = nn.Linear(512, in_dim)
        self.embedding_text = nn.Linear(512, in_dim)

        self.text_enhance = CrossAttention(in_dim, reduce=False)
        self.cross_att_rgb = CrossAttention(in_dim, reduce=False)
        self.spatial_enhance = CrossAttention(in_dim, reduce=True)
        self.temporal_enhance1 = CrossAttention(in_dim, reduce=False)
        self.temporal_enhance2 = CrossAttention(in_dim, reduce=False)

        self.grading_regression = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_class)
        )

        # etf
        self.no_etf = no_etf
        self.num_class = num_class
        etf_vec1, etf_rect = eth(self.num_class, self.num_class) 
        self.register_buffer('etf_vec1', etf_vec1)
        self.etf_rect = etf_rect


    def forward(self, rgb, flow, skeleton, CLIPvideo, text):
        rgb_parts = torch.split(rgb, 1, dim=1)
        flow_parts = torch.split(flow, 1, dim=1)
        skeleton_parts = torch.split(skeleton, 1, dim=1)

        if CLIPvideo.shape[-1] == self.embedding_CLIPvideo.in_features:
            CLIPvideo = self.embedding_CLIPvideo(CLIPvideo)
        
        if text.shape[-1] == self.embedding_text.in_features:
            text = self.embedding_text(text)

        temporal_out = torch.tensor([]).to(device)

        for rgb_part, flow_part, skeleton_part in zip(rgb_parts, flow_parts, skeleton_parts):
            rgb_part = rgb_part.squeeze(1)
            flow_part = flow_part.squeeze(1)
            skeleton_part = skeleton_part.squeeze(1)

            if rgb_part.shape[-1] == self.embedding_rgb.in_features:
                rgb_part = self.embedding_rgb(rgb_part)
            
            out = self.text_enhance(CLIPvideo, text, text) # [Q, K, V] => [(CLIPvideo), rext, text]
            out1 = self.cross_att_rgb(out, rgb_part, rgb_part) # [Q, K, V] => [(out), (rgb), (rgb)]
            
            out = self.spatial_enhance(skeleton_part, torch.cat((rgb_part, flow_part), dim=1), torch.cat((rgb_part, flow_part), dim=1)) # [Q, K, V] => [(ske), (rgb, flow), (rgb, flow)] 
            out2 = self.temporal_enhance1(out, rgb_part, rgb_part) # [Q, K, V] => [(out), (rgb), (rgb)]
            out3 = self.temporal_enhance2(out, flow_part, flow_part) # [Q, K, V] => [(out), (flow), (flow)]
            out = torch.cat((out1, out2, out3), dim=1).unsqueeze(1) # [B, 1024] * 3 => [B, 1, 3072]
            temporal_out = torch.cat((temporal_out, out), dim=1)    # [B, 10, 3072]
        
        out = temporal_out.mean(dim=1)  # [B, 10, 3072] => [B, 3072]
        out_embedding = out
        # print("output decoder concat is ", out.shape)

        # regressor
        out = self.grading_regression(out)

        if self.no_etf:
            out_logits = out
        else:
            out_logits = torch.matmul(out, self.etf_vec1)

        out = F.softmax(out_logits, dim=-1)
        score = torch.argmax(out, dim=-1)

        output = {
            'embed': out_embedding,
            'out_softmax': out
        }
        # print("out decoder is {}, and output is {}".format(out.shape, out_logits))

        return output, out_logits, score

