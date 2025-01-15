import os
import time
from datetime import datetime
from scipy.stats import spearmanr
from collections import OrderedDict
import numpy as np
import einops
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

from data.dataset import get_dataloader
from models.i3d import I3D
from models.video_swin_transformer import SwinTransformer3D
from models.GCN import GCN
from models.loss import LossCalculator
from models.model import DualAlign

from utils.utils import load_config
from utils.utils import setup_logger
from utils.utils import check_cuda

def test_epoch(args, epoch, test_loader, rgb_model, I3D_flow, GCN_skeleton, model, best_acc):
    if not args.pretrain_feats:
        rgb_model.eval()
        I3D_flow.eval()
    GCN_skeleton.eval()
    model.eval()

    scores = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for batch_idx, (rgb, flow, skeleton, CLIPvideo, text, label) in enumerate(tqdm(test_loader, desc=f"Testing Epoch {epoch+1}")):
            rgb = rgb.float().cuda()
            flow = flow.float().cuda()
            skeleton = skeleton.float().cuda()
            # 10 clips, each clip contains 16 frames
            start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]

            if args.pretrain_feats:
                # rgb [B, 10, 768] flow [B, 10, 1024]
                rgb_feature = rgb
                flow_feature = flow
            else:
                rgb_clips = torch.cat([rgb[:, :, i: i + 16] for i in start_idx])
                flow_clips = torch.cat([flow[:, :, i: i + 16] for i in start_idx])

                if args.backbone == 'VST':
                    rgb_outputs_logits = rgb_model(rgb)
                    # rgb_outputs_logits shape: [1, 1024, 52, 7, 7], average pooling to [1, 1024]
                    rgb_outputs = rgb_outputs_logits.mean(2)
                    rgb_outputs = rgb_outputs.mean(2)
                    rgb_outputs = rgb_outputs.mean(2)
                else:
                    rgb_outputs, rgb_feature = rgb_model(rgb_clips)
                
                flow_outputs, flow_feature = I3D_flow(flow_clips)

                # Adjust feature dimensions
                # Reshape rgb_feature, flow_feature from [10, 1024, 1, 1, 1] to [1, 10, 1024]
                rgb_feature = rgb_feature.squeeze(2).squeeze(2).squeeze(2)
                flow_feature = flow_feature.squeeze(2).squeeze(2).squeeze(2)

            # Preserve Batch dimension, reshape from [B, 103, 26, 3] to [B, 10, 16, 26, 3]
            skeleton = skeleton.unsqueeze(dim=1)
            skeleton_clips = torch.cat([skeleton[:, :, i: i + 16] for i in start_idx], dim=1)

            # Reshape skeleton_clips from [B, 10, 16, 26, 3] to [B*10, 16, 26, 3], process, then reshape back
            skeleton_clips = einops.rearrange(skeleton_clips, 'B S T N C -> (B S) T N C')
            skeleton_feature = GCN_skeleton(skeleton_clips)
            skeleton_feature = einops.rearrange(skeleton_feature, '(B S) C -> B S C', B=skeleton.shape[0])

            # Reshape CLIPvideo from [1, 103, 512] to [1, 512] using average pooling to represent frame-level features
            CLIPvideo = CLIPvideo.mean(1)
            # Reshape text from [1, 1, 512] to [1, 512]
            text = text.squeeze(1).float()  # Convert from HalfTensor to FloatTensor

            output, _, score = model(rgb_feature, flow_feature, skeleton_feature, CLIPvideo, text)

            out_sm = output['out_softmax']

            if len(scores) == 0:
                scores = score.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
                out_sms = out_sm.cpu().detach().numpy()
            else:
                scores = np.concatenate((scores, score.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
                out_sms = np.concatenate((out_sms, out_sm.cpu().detach().numpy()), axis=0)

    avg_coef, _ = spearmanr(scores, labels)
    infer_result = {
        'labels': labels,
        'scores': scores,
        'out_sms': out_sms
    }

    pred_scores = scores.reshape(-1,)
    true_scores = labels.reshape(-1,)

    # Calculate prediction accuracy given pred and true scores
    accuracy = np.mean(np.abs(pred_scores - true_scores) <= 0.01)

    rl2 = 100 * np.power((pred_scores - true_scores) /
                       (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
    
    return avg_coef, rl2, accuracy, infer_result


def train(args):
    # If basic.log exists, rename it based on the number of .log files in the folder to avoid overwriting
    if os.path.exists(os.path.join(args.save_path, args.exp_name, 'basic.log')):
        file_count = len([name for name in os.listdir(os.path.join(args.save_path, args.exp_name)) if name.endswith('.log')])
        file_name = os.path.join(args.save_path, args.exp_name, f'basic_{file_count}.log')     
        logger = setup_logger('basic_logger', file_name, level=logging.INFO)
    else:
        logger = setup_logger('basic_logger', os.path.join(args.save_path, args.exp_name, 'basic.log'), level=logging.INFO)
    logger.info("Start training with the following configurations:")
    
    # Print all configurations from args to the log file
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    # Data loading and preprocessing
    transform = transforms.Compose([
            transforms.Resize((455,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    train_loader = get_dataloader(args, is_train=True, transform=transform)
    test_loader = get_dataloader(args, is_train=False, transform=transform, shuffle=False)

    # Model, loss function, optimizer
    # Load pre-trained I3D model
    if args.pretrain_feats:
        print("Using pretrained V & O features.")
    else:
        if args.backbone == 'VST':
            rgb_model = SwinTransformer3D(patch_size=(2,4,4),
                            embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=(8,7,7),
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            drop_path_rate=0.2,
                            patch_norm=True)
            new_state_dict = OrderedDict()
            checkpoint = torch.load(args.VST_rgb_model_path)
            for k, v in checkpoint['state_dict'].items():
                if 'backbone' in k:
                    name = k[9:]
                    new_state_dict[name] = v 
            rgb_model.load_state_dict(new_state_dict)
            rgb_model = rgb_model.cuda()
            print("VST rgb model loaded successfully.")
        else:
            rgb_model = I3D(num_classes=args.I3D_pretrained_num_classes, modality='rgb')
            rgb_model.load_state_dict(torch.load(args.I3D_rgb_model_path))
            rgb_model = rgb_model.cuda()
            print("I3D rgb model loaded successfully.")

        I3D_flow = I3D(num_classes=args.I3D_pretrained_num_classes, modality='flow')
        I3D_flow.load_state_dict(torch.load(args.I3D_flow_model_path))
        I3D_flow = I3D_flow.cuda()
        print("I3D flow model loaded successfully.")

    # Check GPU memory usage
    # check_cuda()

    GCN_skeleton = GCN(num_nodes=args.num_nodes, in_features=args.in_features, hidden_features=args.hidden_features, out_features=args.out_features)
    GCN_skeleton = GCN_skeleton.cuda()

    model = DualAlign(args.in_dim, args.hidden_dim, args.num_classes, args.n_head, args.n_query, args.dropout, args.no_etf)
    model = model.cuda()

    # Loss function for calculating cosine similarity
    LossFun = LossCalculator(args, alpha=1, margin=0.5, num_classes=args.num_classes, etf_head=False, loss_align=0)

    best_acc = 0.0
    best_coef = 0.0
    best_rmse = 100.0
    last_best_file = ""

    # Set up optimizer, optimize parameters of GCN_skeleton and model, with weight decay and learning rate decay
    if args.pretrain_feats:
        optimizer = optim.Adam([
            {'params': GCN_skeleton.parameters()},
            {'params': model.parameters()}
        ], lr=args.learning_rate, eps=1e-8, weight_decay=1e-4)
    else:
        optimizer = optim.Adam([
            {'params': rgb_model.parameters()},
            {'params': I3D_flow.parameters()},
            {'params': GCN_skeleton.parameters()},
            {'params': model.parameters()}
        ], lr=args.learning_rate, eps=1e-8, weight_decay=1e-4)


    # Define warmup and total steps
    warmup_steps = 500
    total_steps = 10000

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    # Create LambdaLR scheduler for warmup
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Create CosineAnnealingLR scheduler for cosine annealing
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    
    # Check if saved models exist
    start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # If pretrain_feats is True, no need to train VST model
        if not args.pretrain_feats:
            rgb_model.train()
            I3D_flow.train()
        GCN_skeleton.train()
        model.train()
        running_loss = 0.0
        all_feats = torch.Tensor().cuda()
        for batch_idx, (rgb, flow, skeleton, CLIPvideo, text, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            rgb = rgb.float().cuda()
            flow = flow.float().cuda()
            skeleton = skeleton.float().cuda()
            # To reduce GPU memory usage, split the temporal data into 10 clips, each containing 16 frames
            start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]

            if args.pretrain_feats:
                # rgb [B, 10, 768] flow [B, 10, 1024]
                rgb_feature = rgb
                flow_feature = flow
            else:
                rgb_clips = torch.cat([rgb[:, :, i: i + 16] for i in start_idx])
                flow_clips = torch.cat([flow[:, :, i: i + 16] for i in start_idx])

                if args.backbone == 'VST':
                    rgb_outputs_logits = rgb_model(rgb)
                    # rgb_outputs_logits shape: [1, 1024, 52, 7, 7], average pooling to [1, 1024]
                    rgb_outputs = rgb_outputs_logits.mean(2)
                    rgb_outputs = rgb_outputs.mean(2)
                    rgb_outputs = rgb_outputs.mean(2)
                else:
                    rgb_outputs, rgb_feature = rgb_model(rgb_clips)
                
                flow_outputs, flow_feature = I3D_flow(flow_clips)

                # Adjust feature dimensions
                # Reshape rgb_feature, flow_feature from [10, 1024, 1, 1, 1] to [1, 10, 1024]
                rgb_feature = rgb_feature.squeeze(2).squeeze(2).squeeze(2)
                flow_feature = flow_feature.squeeze(2).squeeze(2).squeeze(2)
            
            # Preserve Batch dimension, reshape from [B, 103, 26, 3] to [B, 10, 16, 26, 3]
            skeleton = skeleton.unsqueeze(dim=1)
            skeleton_clips = torch.cat([skeleton[:, :, i: i + 16] for i in start_idx], dim=1)

            # Reshape skeleton_clips from [B, 10, 16, 26, 3] to [B*10, 16, 26, 3], process, then reshape back
            skeleton_clips = einops.rearrange(skeleton_clips, 'B S T N C -> (B S) T N C')
            skeleton_feature = GCN_skeleton(skeleton_clips)
            skeleton_feature = einops.rearrange(skeleton_feature, '(B S) C -> B S C', B=skeleton.shape[0])
            
            # Reshape CLIPvideo from [1, 103, 512] to [1, 512] using average pooling to represent frame-level features
            CLIPvideo = CLIPvideo.mean(1)
            # Reshape text from [1, 1, 512] to [1, 512]
            text = text.squeeze(1).float()  # Convert from Half to Float

            raw_feats = [rgb_feature, flow_feature, skeleton_feature, CLIPvideo, text]

            output, out_logits, _ = model(rgb_feature, flow_feature, skeleton_feature, CLIPvideo, text)
            out_logits = out_logits.cpu()  # Note the role of detach
            loss, celoss, maxloss = LossFun(out_logits, labels, output['embed'], raw_feats)

            if args.save_feat:
                if args.pretrain_feats:
                    # VST, I3D, GCN, CLIP, text, embed, labels -> [1, 768+1024+1024+512+512+3072+1] -> [1, 6913]
                    all_feat = torch.cat((rgb_feature.mean(1), flow_feature.mean(1), skeleton_feature.mean(1), \
                                    CLIPvideo, text, output['embed'], labels.unsqueeze(-1).float().cuda()), dim=1)
                else:
                    # Store intermediate features [rgb, flow, skeleton, CLIPvideo, text, output['embed'], labels] -> [1, 1024+1024+1024+512+512+3072+1] -> [1, 7169]
                    all_feat = torch.cat((rgb_feature.mean(0,keepdim=True), flow_feature.mean(0,keepdim=True), skeleton_feature.mean(0,keepdim=True), \
                                    CLIPvideo, text, output['embed'], labels.unsqueeze(-1).float().cuda()), dim=1)
                all_feats = torch.cat((all_feats, all_feat), dim=0)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate
            if warmup_scheduler.last_epoch < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step(epoch - warmup_steps)
            
            running_loss += loss.item()

        # After each epoch, scheduler step, save the model
        # scheduler.step()
        if args.pretrain_feats:
            avg_coef, rl2, acc, results = test_epoch(args, epoch, test_loader, None, None, GCN_skeleton, model, best_acc)
        else:
            avg_coef, rl2, acc, results = test_epoch(args, epoch, test_loader, rgb_model, I3D_flow, GCN_skeleton, model, best_acc)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Coef: {avg_coef:.4f}, Acc: {acc:.4f}, RL2: {rl2:.4f}, Average Loss: {running_loss/len(train_loader):.4f}")
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Average Coef: {avg_coef:.4f}, Acc: {acc:.4f}, RL2: {rl2:.4f}, Average Loss: {running_loss/len(train_loader):.4f}")

        # Save the model based on accuracy
        if acc > best_acc or avg_coef > best_coef or rl2 < best_rmse:
            # Find and delete the previous best model
            if os.path.exists(last_best_file):
                os.remove(last_best_file)
            if acc > best_acc:
                best_acc = acc
            if avg_coef > best_coef:
                best_coef = avg_coef
            if rl2 < best_rmse:
                best_rmse = rl2
            if args.save_feat:
                cur_file_name = os.path.join(args.save_path, args.exp_name, f"E{epoch + 1}_Ac{best_acc*100:.2f}_Co{best_coef*100:.2f}_F.pth")
                last_best_file = cur_file_name
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'all_feats': all_feats,
                    'results': results,
                    'loss': running_loss / len(train_loader),
                }, os.path.join(args.save_path, args.exp_name, f"E{epoch + 1}_Ac{best_acc*100:.2f}_Co{best_coef*100:.2f}_F.pth")) 
            else:
                cur_file_name = os.path.join(args.save_path, args.exp_name, f"E{epoch + 1}_Ac{best_acc*100:.2f}_Co{best_coef*100:.2f}.pth")
                last_best_file = cur_file_name
                torch.save({
                    'epoch': epoch,
                    'loss': running_loss / len(train_loader),
                }, cur_file_name)
            logger.info(f"Model saved at epoch {epoch+1} with best acc {best_acc:.4f}, coef {best_coef:.4f}")

    logger.info("Training finished.")
    logger.info(f"Best Acc: {best_acc:.4f}, Best Coef: {best_coef:.4f} at epoch {epoch+1}")

    # Write the overall results of this training to results.csv
    with open(os.path.join('exps', 'results.csv'), 'a') as f:
        # If the file is empty, write the header
        if os.path.getsize(os.path.join('exps', 'results.csv')) == 0:
            f.write("Time,Action,ExpName,BestCoef,BestRMSE,BestAcc\n")
        # Write the current time, accurate to seconds
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},")
        f.write(f"{args.action},{args.exp_name},{best_coef:.4f},{best_rmse:.4f},{best_acc:.4f}\n")


if __name__ == "__main__":
    train()