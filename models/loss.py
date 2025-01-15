import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MultiModalAlignmentLoss(nn.Module):
    def __init__(self, tau: float = 0.1):
        """
        Initialize the Multi-Modal Alignment Loss module.

        Args:
            tau (float): Temperature parameter for scaling.
        """
        super(MultiModalAlignmentLoss, self).__init__()
        self.tau = tau

    @staticmethod
    def normalize_vectors(vectors):
        """
        Normalize vectors to unit norm.

        Args:
            vectors (torch.Tensor): Input vectors of shape (..., D), where D is the feature dimension.

        Returns:
            torch.Tensor: Normalized vectors of the same shape.
        """
        return F.normalize(vectors, p=2, dim=-1)

    @staticmethod
    def gram_matrix(vectors):
        """
        Compute the Gram matrix from modality vectors.

        Args:
            vectors (torch.Tensor): Modality vectors of shape (K, D),
                                    where K is the number of modalities and D is the feature dimension.

        Returns:
            torch.Tensor: Gram matrix of shape (K, K).
        """
        return torch.matmul(vectors, vectors.T)

    @staticmethod
    def gramian_volume(gram_matrix):
        """
        Compute the Gramian volume as the square root of the determinant of the Gram matrix.

        Args:
            gram_matrix (torch.Tensor): Gram matrix of shape (K, K).

        Returns:
            torch.Tensor: The Gramian volume.
        """
        det = torch.det(gram_matrix)
        return torch.sqrt(torch.clamp(det, min=1e-12))  # Stability clamp to avoid sqrt of negative or zero.

    def compute_loss(self, anchors, modalities):
        """
        Compute the direct-anchor and anchor-direct losses.

        Args:
            anchors (torch.Tensor): Anchor vectors of shape (B, D), where B is the batch size and D is the feature dimension.
            modalities (torch.Tensor): Modality vectors of shape (B, K, D),
                                       where K is the number of modalities.

        Returns:
            torch.Tensor: Combined multi-modal alignment loss.
        """
        batch_size, K, D = modalities.shape
        l_da = 0.0
        l_ad = 0.0

        for i in range(batch_size):
            ai = anchors[i]  # Anchor vector (D,)
            mi = modalities[i]  # Modality vectors (K, D)

            # Normalize the anchor and modality vectors
            ai = self.normalize_vectors(ai)
            mi = self.normalize_vectors(mi)

            # Combine anchor and modality vectors for volume calculation
            combined_da = torch.cat((ai.unsqueeze(0), mi), dim=0)
            volume_ai = self.gramian_volume(self.gram_matrix(combined_da))

            # Direct-anchor loss numerator
            numerator_da = torch.exp(-volume_ai / self.tau)

            # Direct-anchor loss denominator
            denominator_da = 0.0
            for j in range(batch_size):
                aj = anchors[j]
                mj = modalities[j]
                aj = self.normalize_vectors(aj)
                mj = self.normalize_vectors(mj)

                combined = torch.cat((aj.unsqueeze(0), mi), dim=0)
                volume_aj = self.gramian_volume(self.gram_matrix(combined))
                denominator_da += torch.exp(-volume_aj / self.tau)

            l_da -= torch.log(numerator_da / denominator_da)

            # Anchor-direct loss numerator
            numerator_ad = torch.exp(-volume_ai / self.tau)

            # Anchor-direct loss denominator
            denominator_ad = 0.0
            for j in range(batch_size):
                aj = anchors[j]
                mj = modalities[j]
                aj = self.normalize_vectors(aj)
                mj = self.normalize_vectors(mj)

                combined = torch.cat((ai.unsqueeze(0), mj), dim=0)
                volume_aj = self.gramian_volume(self.gram_matrix(combined))
                denominator_ad += torch.exp(-volume_aj / self.tau)

            l_ad -= torch.log(numerator_ad / denominator_ad)

        # Average over batch size
        l_da = l_da / batch_size
        l_ad = l_ad / batch_size

        # Combine the two losses
        return l_da + l_ad

    def forward(self, anchors, modalities):
        """
        Forward pass to compute the loss.

        Args:
            anchors (torch.Tensor): Anchor vectors of shape (B, D), where B is the batch size and D is the feature dimension.
            modalities (torch.Tensor): Modality vectors of shape (B, K, D),
                                       where K is the number of modalities.

        Returns:
            torch.Tensor: Multi-modal alignment loss.
        """
        return self.compute_loss(anchors, modalities)

class CosineSimilarityAlignmentLoss(torch.nn.Module):
    def __init__(self, tau: float = 0.1):
        """
        Initialize the Cosine Similarity Alignment Loss module.

        Args:
            tau (float): Temperature parameter for scaling.
        """
        super(CosineSimilarityAlignmentLoss, self).__init__()
        self.tau = tau

    @staticmethod
    def normalize_vectors(vectors):
        """
        Normalize vectors to unit norm.

        Args:
            vectors (torch.Tensor): Input vectors of shape (..., D), where D is the feature dimension.

        Returns:
            torch.Tensor: Normalized vectors of the same shape.
        """
        return F.normalize(vectors, p=2, dim=-1)

    def cosine_similarity(self, v1, v2):
        """
        Compute the cosine similarity between two vectors.

        Args:
            v1 (torch.Tensor): Vector 1 of shape (D,).
            v2 (torch.Tensor): Vector 2 of shape (D,).

        Returns:
            torch.Tensor: Cosine similarity (scalar).
        """
        return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-12)

    def compute_loss(self, anchors, modalities):
        """
        Compute the direct-anchor and anchor-direct losses.

        Args:
            anchors (torch.Tensor): Anchor vectors of shape (B, D), where B is the batch size and D is the feature dimension.
            modalities (torch.Tensor): Modality vectors of shape (B, K, D),
                                       where K is the number of modalities.

        Returns:
            torch.Tensor: Combined alignment loss based on cosine similarity.
        """
        batch_size, K, D = modalities.shape
        l_da = 0.0
        l_ad = 0.0

        for i in range(batch_size):
            ai = anchors[i]  # Anchor vector (D,)
            mi = modalities[i]  # Modality vectors (K, D)

            # Normalize the anchor and modality vectors
            ai = self.normalize_vectors(ai)
            mi = self.normalize_vectors(mi)

            # Direct-anchor loss numerator
            numerator_da = torch.exp(
                torch.mean(torch.stack([self.cosine_similarity(ai, m) for m in mi])) / self.tau
            )

            # Direct-anchor loss denominator
            denominator_da = 0.0
            for j in range(batch_size):
                aj = anchors[j]
                mj = modalities[j]
                aj = self.normalize_vectors(aj)
                mj = self.normalize_vectors(mj)

                denominator_da += torch.exp(
                    torch.mean(torch.stack([self.cosine_similarity(aj, m) for m in mi])) / self.tau
                )

            l_da -= torch.log(numerator_da / denominator_da)

            # Anchor-direct loss numerator
            numerator_ad = torch.exp(
                torch.mean(torch.stack([self.cosine_similarity(ai, m) for m in mi])) / self.tau
            )

            # Anchor-direct loss denominator
            denominator_ad = 0.0
            for j in range(batch_size):
                aj = anchors[j]
                mj = modalities[j]
                aj = self.normalize_vectors(aj)
                mj = self.normalize_vectors(mj)

                denominator_ad += torch.exp(
                    torch.mean(torch.stack([self.cosine_similarity(ai, m) for m in mj])) / self.tau
                )

            l_ad -= torch.log(numerator_ad / denominator_ad)

        # Average over batch size
        l_da = l_da / batch_size
        l_ad = l_ad / batch_size

        # Combine the two losses
        return l_da + l_ad

    def forward(self, anchors, modalities):
        """
        Forward pass to compute the loss.

        Args:
            anchors (torch.Tensor): Anchor vectors of shape (B, D), where B is the batch size and D is the feature dimension.
            modalities (torch.Tensor): Modality vectors of shape (B, K, D),
                                       where K is the number of modalities.

        Returns:
            torch.Tensor: Multi-modal alignment loss.
        """
        return self.compute_loss(anchors, modalities)
    

# Example Usage
if __name__ == "__main__":
    batch_size = 4
    num_modalities = 3
    feature_dim = 1024
    tau = 0.1

    # Random inputs
    anchors = torch.randn(batch_size, feature_dim)
    modalities = torch.randn(batch_size, num_modalities, feature_dim)

    # Initialize loss function
    loss_fn = MultiModalAlignmentLoss(tau=tau)

    # Compute loss
    loss = loss_fn(anchors, modalities)
    print("Loss:", loss.item())

 
class LossCalculator(nn.Module):
    def __init__(self, args, alpha, margin, num_classes, etf_head = False, loss_align = 0):
        super(LossCalculator, self).__init__()
        self.MSE_loss = nn.MSELoss()
        self.CE_loss = nn.CrossEntropyLoss()
        self.Cos_loss = nn.CosineEmbeddingLoss()
        self.MMA_loss = MultiModalAlignmentLoss(tau=0.1)
        self.CSA_loss = CosineSimilarityAlignmentLoss(tau=0.1)
        self.alpha = alpha
        self.beta = 0.5
        self.num_classes = num_classes
        self.lossCalType = args.lossType if hasattr(args, 'lossType') else 'Normal'
        # S stands for stage from 0 to 2, M stands for modality from 0 to 4
        # 1S4M: 1 stage, 4 modality align
        # 2S3M2M: 2 stages, 3 modality align in stage 1, 2 modality align in stage 2
        # 1S3M: 1 stage, 3 modality align in stage 1, 2 modality concat
        # 1S2M: 1 stage, 3 modality concat, 2 modality align
        # 0S4M: 4 modality concat
        # Normal: Raw with maximum loss

    def mmAlignment_loss(self, anchors, modalities):
        return self.MMA_loss(anchors, modalities)
    
    def csAlignment_loss(self, anchors, modalities):
        return self.CSA_loss(anchors, modalities)

    def maximizing_loss(self, embedding1, embedding2, embedding3, threshold=0.5):
        similarity12 = torch.sum(embedding1 * embedding2, dim=1) / (torch.norm(embedding1, p=2, dim=1) * torch.norm(embedding2, p=2, dim=1))
        similarity13 = torch.sum(embedding1 * embedding3, dim=1) / (torch.norm(embedding1, p=2, dim=1) * torch.norm(embedding3, p=2, dim=1))
        similarity23 = torch.sum(embedding2 * embedding3, dim=1) / (torch.norm(embedding2, p=2, dim=1) * torch.norm(embedding3, p=2, dim=1))
        loss = torch.mean(torch.relu(similarity12 - threshold)) + torch.mean(torch.relu(similarity13 - threshold)) + torch.mean(torch.relu(similarity23 - threshold))
        return loss

    def custom_label_smoothing(self, pred, target, smoothing=0.1):
        """
        Apply custom label smoothing to the predictions.

        Args:
            pred (torch.Tensor): Model predictions of shape [batch_size, num_classes].
            target (torch.Tensor): Ground truth labels of shape [batch_size].
            smoothing (float): Smoothing factor (default: 0.1).

        Returns:
            torch.Tensor: Smoothed loss value.
        """
        confidence = 1.0 - smoothing  # Confidence for the true class
        true_dist = torch.zeros_like(pred)  # Initialize the smoothed target distribution
        smoothing_value = smoothing / 2  # Noise value to distribute to neighboring classes

        for i in range(target.size(0)):  # Iterate over each sample in the batch
            true_dist[i].fill_(0)  # Initialize the distribution for the current sample to 0
            true_dist[i][target[i]] = confidence  # Assign confidence to the true class

            # Distribute smoothing noise to neighboring classes
            if target[i] > 0 and target[i] < self.num_classes - 1:
                # For non-boundary classes, distribute noise to both left and right neighbors
                true_dist[i][target[i] - 1] = smoothing_value  # Left neighbor
                true_dist[i][target[i] + 1] = smoothing_value  # Right neighbor
            elif target[i] == 0:
                # For the first class (boundary case), distribute all noise to the right neighbor
                true_dist[i][target[i] + 1] = smoothing
            elif target[i] == self.num_classes - 1:
                # For the last class (boundary case), distribute all noise to the left neighbor
                true_dist[i][target[i] - 1] = smoothing

        # Compute the cross-entropy loss between the smoothed target distribution and the predictions
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))


    def forward(self, pred, label, feat, raw_feats):
        # feat [B, 3*C] -> embedding1 [B, C] + embedding2 [B, C] + embedding3 [B, C]
        embedding1 = feat[:, :feat.shape[1]//3]
        embedding2 = feat[:, feat.shape[1]//3:2*feat.shape[1]//3]
        embedding3 = feat[:, 2*feat.shape[1]//3:]
        # CLIP_video [B, 0.5C] text [B, 0.5C] -> CLIP[B, C]
        clip = torch.cat([raw_feats[3], raw_feats[4]], dim=1)
        
        max_loss = self.maximizing_loss(embedding1, embedding2, embedding3)

        if self.lossCalType == 'woLabel' and (label == -1).any():
            # embedding2 [B, C] embedding3 [B, C] -> modalities [B, 2, C]
            modalities = torch.stack([embedding2, embedding3], dim=1)
            mmAlign_loss1 = self.mmAlignment_loss(embedding1, modalities)
            mmAlign_loss2 = self.mmAlignment_loss(embedding1, clip.unsqueeze(1))
            loss = self.alpha * (mmAlign_loss1 + mmAlign_loss2)
            return loss, 0, 0

        celoss = self.CE_loss(pred, label)
        label_smoothing_loss = self.custom_label_smoothing(pred, label, smoothing=0.1)

        if self.lossCalType == '1S4M':
            # embedding2 [B, C] embedding3 [B, C] clip [B, C] -> modalities [B, 3, C]
            modalities = torch.stack([embedding2, embedding3, clip], dim=1)
            mmAlign_loss = self.mmAlignment_loss(embedding1, modalities)
            loss = self.beta * (celoss + label_smoothing_loss) + self.alpha * mmAlign_loss
        elif self.lossCalType == '1S3M':
            # embedding2 [B, C] embedding3 [B, C] -> modalities [B, 2, C]
            modalities = torch.stack([embedding2, embedding3], dim=1)
            mmAlign_loss = self.mmAlignment_loss(embedding1, modalities)
            loss = self.beta * (celoss + label_smoothing_loss) + self.alpha * mmAlign_loss
        elif self.lossCalType == '2S3M2M' or self.lossCalType == 'woLabel':
            # embedding2 [B, C] embedding3 [B, C] -> modalities [B, 2, C]
            modalities = torch.stack([embedding2, embedding3], dim=1)
            mmAlign_loss1 = self.mmAlignment_loss(embedding1, modalities)
            mmAlign_loss2 = self.mmAlignment_loss(embedding1, clip.unsqueeze(1))
            loss = self.beta * (celoss + label_smoothing_loss) + self.alpha * (mmAlign_loss1 + mmAlign_loss2)
        elif self.lossCalType == '1S2M':
            mmAlign_loss = self.mmAlignment_loss(embedding1, clip.unsqueeze(1))
            loss = self.beta * (celoss + label_smoothing_loss) + self.alpha * mmAlign_loss
        elif self.lossCalType == 'woGram': # denote 2S3M2M with Cosine Similarity alignmentloss
            # embedding2 [B, C] embedding3 [B, C] -> modalities [B, 2, C]
            modalities = torch.stack([embedding2, embedding3], dim=1)
            csAlign_loss1 = self.csAlignment_loss(embedding1, modalities)
            csAlign_loss2 = self.csAlignment_loss(embedding1, clip.unsqueeze(1))
            loss = self.beta * (celoss + label_smoothing_loss) + self.alpha * (csAlign_loss1 + csAlign_loss2)
        elif self.lossCalType == 'woAlign':
            loss = self.beta * (celoss + label_smoothing_loss)
        else:
            loss = self.beta * (celoss + label_smoothing_loss) + self.alpha * max_loss
     
        return loss, celoss, max_loss
      
