import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, reduce, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        
        # 线性变换，将拼接后的特征维度映射回 feature_dim
        self.reduce = reduce
        self.reduce_dim = nn.Linear(feature_dim * 2, feature_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # print("attention shape is ", q.shape, k.shape, v.shape)

        B = q.shape[0]

        # 拼接后特征维度变为 2 * feature_dim，需要重新映射回 feature_dim
        if self.reduce:
            k = self.reduce_dim(k)
            v = self.reduce_dim(v)
        
        # Linear transformations
        Q = self.query(q)  # [B, C] -> [B, C]
        K = self.key(k)    # [B, C] -> [B, C]
        V = self.value(v)  # [B, C] -> [B, C]

        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim).permute(1, 0, 2)  # [B, C] -> [num_heads, B, head_dim]
        K = K.view(B, self.num_heads, self.head_dim).permute(1, 0, 2)  # [B, C] -> [num_heads, B, head_dim]
        V = V.view(B, self.num_heads, self.head_dim).permute(1, 0, 2)  # [B, C] -> [num_heads, B, head_dim]

        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / (self.head_dim ** 0.5)  # [num_heads, B, head_dim] x [num_heads, head_dim, B] -> [num_heads, B, B]
        attention = torch.softmax(energy, dim=-1)  # [num_heads, B, B]
        attention = self.dropout(attention)  # Apply dropout to the attention scores
        out = torch.matmul(attention, V)  # [num_heads, B, B] x [num_heads, B, head_dim] -> [num_heads, B, head_dim]

        # Concatenate heads
        out = out.permute(1, 0, 2).contiguous().view(B, self.feature_dim)  # [num_heads, B, head_dim] -> [B, C]

        # Apply dropout to the value before the final linear transformation
        out = self.dropout(out)

        # Final linear transformation
        out = self.fc_out(out)  # [B, C] -> [B, C]

        return out

if __name__ == "__main__":
    # 示例数据
    batch_size = 32
    feature_dim = 1024

    # 假设输入数据的格式为 [batch_size, feature_dimension]
    data1 = torch.randn(batch_size, feature_dim)
    data2 = torch.randn(batch_size, feature_dim)
    data3 = torch.randn(batch_size, feature_dim)

    # 初始化模型
    model = CrossAttention(feature_dim, reduce=True)

    # 计算嵌入表示
    # 在这个示例中，我们将 data1 作为 Q，data2 和 data3 拼接在一起作为 K 和 V
    K = torch.cat((data2, data3), dim=1)
    V = torch.cat((data2, data3), dim=1)
    output = model(data1, K, V)

    # 打印输出的形状
    print("output shape:", output.shape)
