import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Define a simple GCN model for fraud detection
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Function for training the model
def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    
    # Get the embeddings for the nodes involved in each edge (user-merchant)
    edge_index = data.edge_index
    edge_embeddings = out[edge_index[0]] * out[edge_index[1]]  # Element-wise multiplication of node features
    
    # Binary cross-entropy loss (target must be binary labels for links)
    target = data.y
    loss = F.binary_cross_entropy_with_logits(edge_embeddings, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# Function to get the top-k potential fraudulent transactions
def get_top_links(model, data, top_k=3):
    model.eval()
    with torch.no_grad():
        out = model(data)

    # Get embeddings for each edge (user-merchant)
    edge_index = data.edge_index
    edge_embeddings = out[edge_index[0]] * out[edge_index[1]]  # Element-wise multiplication of node features
    
    # Get the scores for each link (dot product)
    scores = edge_embeddings.squeeze()
    top_k_indices = scores.topk(top_k).indices

    # Extract the corresponding node pairs for the top K links
    top_k_node_pairs = edge_index[:, top_k_indices]
    
    return top_k_node_pairs

# Load the graph data from a CSV file
def load_graph_data(file_path):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Create the edge_index tensor (user_id, merchant_id)
    edge_index = torch.tensor([df['user_id'].values, df['merchant_id'].values], dtype=torch.long)

    # Create the target labels tensor (is_fraudulent)
    y = torch.tensor(df['is_fraudulent'].values, dtype=torch.float).view(-1, 1)

    # Feature matrix (node features, assuming 1 feature per node)
    num_nodes = df[['user_id', 'merchant_id']].values.max() + 1  # Find the max node index to determine number of nodes
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y), df  # Return df as well

# Define the file path to the CSV
file_path = 'graph_data.csv'

# Load the graph data from the CSV file
data, df = load_graph_data(file_path)

# Initialize the model, optimizer, and training settings
input_dim = 1  # 1 feature per node
hidden_dim = 8  # Hidden layer dimension
output_dim = 1  # Output dimension for edge prediction
model = GNN(input_dim, hidden_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model for 100 epochs
for epoch in range(100):
    loss = train(data, model, optimizer)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Get and display top 3 potential fraudulent transactions
top_k_node_pairs = get_top_links(model, data, top_k=3)

print("Top 3 potential fraudulent transactions (user, merchant):")
print(top_k_node_pairs)
# Visualize the graph using networkx and matplotlib
G = nx.Graph()

# Add edges to the graph (only relationships, no isolated nodes)
for i in range(data.edge_index.shape[1]):
    user_id = data.edge_index[0][i].item()
    merchant_id = data.edge_index[1][i].item()
    G.add_edge(user_id, merchant_id)

# Draw the graph
pos = nx.spring_layout(G)  # Layout for visualization
plt.figure(figsize=(8, 8))

# Color edges based on fraudulent transactions
edge_colors = ['red' if df['is_fraudulent'][i] == 1 else 'gray' for i in range(df.shape[0])]

# Draw the graph with custom edge colors, node_size and color
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=16, font_weight="bold", edge_color=edge_colors)

plt.title("Transaction Network Visualization (Fraudulent Transactions Highlighted)")
plt.show()
