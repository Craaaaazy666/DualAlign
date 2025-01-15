import torch
import torch.nn as nn
import torch.nn.functional as F

# List of edges defining the connections between nodes
l_pair = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
    (17, 18), (18, 19), (19, 11), (19, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (15, 20), (16, 21), (16, 23), (15, 22), (15, 24), (16, 25)  # Foot Type 1
    # (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)  # Foot Type 2 (commented out)
]

# Function to create an adjacency matrix
def create_adjacency_matrix(num_nodes, edges):
    """
    Create an adjacency matrix for a graph given the number of nodes and edges.

    Args:
        num_nodes (int): Number of nodes in the graph.
        edges (list): List of tuples representing edges between nodes.

    Returns:
        torch.Tensor: Adjacency matrix of shape [num_nodes, num_nodes].
    """
    adj_matrix = torch.eye(num_nodes)  # Initialize with identity matrix (self-connections)
    for i, j in edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # Undirected graph, fill symmetric elements
    return adj_matrix

# Calculate the number of nodes
num_nodes = max(max(i, j) for i, j in l_pair) + 1
# Create the adjacency matrix
adj_matrix = create_adjacency_matrix(num_nodes, l_pair)

class GCNLayer(nn.Module):
    """
    A single Graph Convolutional Network (GCN) layer.
    """
    def __init__(self, in_features, out_features):
        """
        Initialize the GCN layer.

        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
        """
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Forward pass for the GCN layer.

        Args:
            x (torch.Tensor): Input tensor of shape [T, N, in_features], where T is the number of frames,
                              N is the number of nodes, and in_features is the input feature dimension.
            adj (torch.Tensor): Adjacency matrix of shape [N, N].

        Returns:
            torch.Tensor: Output tensor of shape [T, N, out_features].
        """
        # Compute degree matrix D
        D = adj.sum(1)
        # Compute normalized adjacency matrix D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        norm_adj = (D_inv_sqrt @ adj @ D_inv_sqrt).float()
        # Apply graph convolution operation
        x = self.linear(x)
        x = F.relu(x)
        x = norm_adj @ x
        return x

class GCN(nn.Module):
    """
    A simple Graph Convolutional Network (GCN) model.
    """
    def __init__(self, num_nodes, in_features, hidden_features, out_features):
        """
        Initialize the SimpleGCN model.

        Args:
            num_nodes (int): Number of nodes in the graph.
            in_features (int): Number of input features per node.
            hidden_features (int): Number of hidden features per node.
            out_features (int): Number of output features.
        """
        super(GCN, self).__init__()
        self.gcn_layer = GCNLayer(in_features, hidden_features)
        self.fc = nn.Linear(hidden_features, out_features)
        self.adj = adj_matrix.cuda()  # Move adjacency matrix to GPU

    def forward(self, x):
        """
        Forward pass for the SimpleGCN model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, N, in_features], where B is the batch size,
                              T is the number of frames, N is the number of nodes, and in_features is the input feature dimension.

        Returns:
            torch.Tensor: Output tensor of shape [B, out_features].
        """
        # Apply GCN layer
        x = self.gcn_layer(x, self.adj)
        # Apply fully connected layer
        x = self.fc(x)
        # Average over frames and nodes to get a fixed-size output
        x = torch.mean(x, dim=1)  # Average over frames
        x = torch.mean(x, dim=1)  # Average over nodes
        return x