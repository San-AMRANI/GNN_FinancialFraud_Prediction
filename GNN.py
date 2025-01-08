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

# Function to get the predictions for a new graph
def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
    
    # Get embeddings for each edge (user-merchant)
    edge_index = data.edge_index
    edge_embeddings = out[edge_index[0]] * out[edge_index[1]]  # Element-wise multiplication of node features
    
    # Get the predicted probabilities for each link
    predictions = torch.sigmoid(edge_embeddings).squeeze()
    
    return predictions

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

# Load the graph data from the training CSV file (graph_data.csv)
train_file_path = 'graph_data.csv'
data, df_train = load_graph_data(train_file_path)

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

# Load the new graph data from the CSV file (new_graph_data.csv)
new_file_path = 'new_graph_data.csv'
df_new = pd.read_csv(new_file_path)

# Create the edge_index for the new graph (user_id, merchant_id)
edge_index_new = torch.tensor([df_new['user_id'].values, df_new['merchant_id'].values], dtype=torch.long)

# Feature matrix for the new graph (same as before, 1 feature per node)
num_nodes_new = df_new[['user_id', 'merchant_id']].values.max() + 1  # Find the max node index to determine number of nodes
x_new = torch.ones((num_nodes_new, 1), dtype=torch.float)

# Prepare the data object for the new graph
data_new = Data(x=x_new, edge_index=edge_index_new)

# Get predictions for the new graph
predictions = predict(model, data_new)

# Add the predictions to the DataFrame
df_new['prediction_prob'] = predictions.numpy()  # Add predicted probabilities to DataFrame
df_new['is_fraudulent'] = predictions.numpy() > 0.5  # Threshold the probabilities (0.5 for fraud detection)

# Calculate the percentage of fraudulent predictions
fraud_percentage = df_new['is_fraudulent'].mean() * 100

# Print the complete version of new_graph_data with the 'is_fraudulent' and 'prediction_prob' columns
print(df_new)

# Print the percentage of fraudulent predictions
print(f"Percentage of fraudulent predictions: {fraud_percentage:.2f}%")

# --- Visualizing the original graph_data ---
G_train = nx.Graph()

# Add edges to the graph (only relationships, no isolated nodes)
for i in range(data.edge_index.shape[1]):
    user_id = data.edge_index[0][i].item()
    merchant_id = data.edge_index[1][i].item()
    G_train.add_edge(user_id, merchant_id)

# Draw the graph for graph_data (original)
plt.figure(figsize=(8, 8))
pos_train = nx.spring_layout(G_train)  # Layout for visualization
edge_colors_train = ['red' if df_train['is_fraudulent'][i] == 1 else 'gray' for i in range(df_train.shape[0])]

# Separate users and merchants
user_nodes_train = set(df_train['user_id'].values)
merchant_nodes_train = set(df_train['merchant_id'].values)

# Assign node colors and sizes
node_colors_train = ['lightblue' if node in user_nodes_train else 'orange' for node in G_train.nodes()]
node_sizes_train = [500 if node in user_nodes_train else 700 for node in G_train.nodes()]

# Draw the graph for the original graph data
nx.draw(G_train, pos_train, with_labels=True, node_size=node_sizes_train, node_color=node_colors_train, font_size=16, font_weight="bold", edge_color=edge_colors_train)
plt.title("Original Transaction Network (graph_data.csv)")
plt.show()

# --- Visualizing the new graph_data (after prediction) ---
G_new = nx.Graph()

# Add edges to the graph (only relationships, no isolated nodes)
for i in range(data_new.edge_index.shape[1]):
    user_id = data_new.edge_index[0][i].item()
    merchant_id = data_new.edge_index[1][i].item()
    G_new.add_edge(user_id, merchant_id)

# Draw the graph for new_graph_data (after prediction)
plt.figure(figsize=(8, 8))
pos_new = nx.spring_layout(G_new)  # Layout for visualization
edge_colors_new = ['red' if df_new['is_fraudulent'][i] == 1 else 'gray' for i in range(df_new.shape[0])]

# Separate users and merchants
user_nodes_new = set(df_new['user_id'].values)
merchant_nodes_new = set(df_new['merchant_id'].values)

# Assign node colors and sizes
node_colors_new = ['lightblue' if node in user_nodes_new else 'orange' for node in G_new.nodes()]
node_sizes_new = [500 if node in user_nodes_new else 700 for node in G_new.nodes()]

# Draw the graph for the predicted graph data
nx.draw(G_new, pos_new, with_labels=True, node_size=node_sizes_new, node_color=node_colors_new, font_size=16, font_weight="bold", edge_color=edge_colors_new)
plt.title("Predicted Transaction Network (new_graph_data.csv after prediction)")
plt.show()
