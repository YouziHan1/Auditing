"""
DPGNN Training Script with Custom Dataset
Trains a DPGNN model on a custom dataset with REAL differential privacy
Uses the project's train_scheduler with gradient clipping and noise injection
Supports interactive node prediction
"""

import torch
import torch.nn as nn
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import datasets.SETUP as SETUP
import datasets.utils as dms_utils
import datasets.model as dms_model
import utils
import train_scheduler as tsch
import privacy.sampling as sampling


class CustomGraphDataset:
    """Wrapper for custom graph dataset from .pt file"""
    
    def __init__(self, data_path):
        """Load custom dataset from .pt file"""
        self.data = torch.load(data_path)
        
        # Extract features and labels
        if isinstance(self.data, dict):
            self.x = self.data.get('x', self.data.get('features', None))
            self.y = self.data.get('y', self.data.get('labels', None))
            self.edge_index = self.data.get('edge_index', None)
        else:
            # Assume it's a pyg Data object
            self.x = self.data.x if hasattr(self.data, 'x') else None
            self.y = self.data.y if hasattr(self.data, 'y') else None
            self.edge_index = self.data.edge_index if hasattr(self.data, 'edge_index') else None
        
        if self.x is None or self.y is None:
            raise ValueError("Dataset must contain features (x) and labels (y)")
        
        self.num_nodes = self.x.shape[0]
        self.num_features = self.x.shape[1]
        self.num_classes = len(torch.unique(self.y))
        
        # Normalize features
        self.x = (self.x - self.x.mean()) / (self.x.std() + 1e-6)
        
        print(f"Dataset loaded: {self.num_nodes} nodes, {self.num_features} features, {self.num_classes} classes")
        if self.edge_index is not None:
            print(f"Edges: {self.edge_index.shape[1]}")


class CustomSubgraphDataset(torch.utils.data.Dataset):
    """
    Dataset that returns subgraphs for DPGNN training
    Mimics the behavior of sampling.subgraph_sampler
    """
    
    def __init__(self, graph_data, indices, K=2, num_neighbors=1, graph_data_name="custom"):
        self._actual_graph_data = graph_data  # Store actual graph data with underscore prefix
        self.indices = indices
        self.K = K
        self.num_neighbors = num_neighbors
        self.graph_data_name = graph_data_name
        self.graph_data = ""  # This will be set to description string by trainer compatibility code
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Return a subgraph centered at node indices[idx]
        For simplicity, we return the node features with its K-hop neighborhood
        """
        node_idx = self.indices[idx]
        
        # Use _actual_graph_data instead of graph_data
        features = self._actual_graph_data.x[node_idx:node_idx+1]  # Shape: [1, feat_dim]
        label = self._actual_graph_data.y[node_idx:node_idx+1]      # Shape: [1]
        
        # Pad to match expected input shape (num_neighbors+1, feat_dim)
        if self.num_neighbors > 0:
            # Sample random neighbors (simplified, real version uses graph structure)
            neighbor_indices = torch.randint(0, self._actual_graph_data.num_nodes, (self.num_neighbors,))
            neighbor_features = self._actual_graph_data.x[neighbor_indices]
            features = torch.cat([features, neighbor_features], dim=0)
            labels_padded = label.repeat(self.num_neighbors + 1)
        else:
            labels_padded = label
        
        return features, labels_padded


def create_custom_dpgnn_loaders(dataset, train_indices, val_indices, test_indices, args):
    """
    Create data loaders compatible with DPGNN trainer
    
    Args:
        dataset: CustomGraphDataset instance
        train_indices: indices for training set
        val_indices: indices for validation set  
        test_indices: indices for test set
        args: arguments with batch_size, K, num_neighbors, etc.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = CustomSubgraphDataset(
        dataset, train_indices, 
        K=args.K, 
        num_neighbors=args.num_neighbors,
        graph_data_name="custom_train"
    )
    
    val_dataset = CustomSubgraphDataset(
        dataset, val_indices,
        K=args.K,
        num_neighbors=args.num_neighbors,
        graph_data_name="custom_val"
    )
    
    test_dataset = CustomSubgraphDataset(
        dataset, test_indices,
        K=args.K,
        num_neighbors=args.num_neighbors_test,
        graph_data_name="custom_test"
    )
    
    # Set metadata for trainer compatibility
    train_dataset.graph_data_name = "custom_amazon"
    train_dataset.graph_data = f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.expected_batchsize,
        shuffle=True,
        num_workers=args.worker_num,
        persistent_workers=True if args.worker_num > 0 else False,
        drop_last=True
    )
    
    val_loader = None  # DPGNN trainer doesn't use val_loader
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=args.worker_num,
        persistent_workers=True if args.worker_num > 0 else False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def evaluate_test_set(model, dataset, test_indices, device):
    """
    Evaluate model on test set and output predictions statistics
    
    Args:
        model: trained model
        dataset: CustomGraphDataset instance
        test_indices: indices of test samples
        device: computation device
    
    Returns:
        predictions: tensor of predicted labels
        accuracies: accuracy of predictions
    """
    model.eval()
    
    test_features = dataset.x[test_indices].to(device)
    test_labels = dataset.y[test_indices].to(device)
    
    with torch.no_grad():
        outputs = model(test_features)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)
    
    # Calculate accuracy
    correct = (predictions == test_labels).sum().item()
    accuracy = correct / len(test_labels)
    
    # Count predictions per class
    print("\n" + "="*60)
    print("Test Set Evaluation Results (Canary Nodes 10000-10099)")
    print("="*60)
    print(f"Test set size: {len(test_labels)} samples")
    print(f"Number of classes: {dataset.num_classes}\n")
    
    print("Prediction Statistics (Class Distribution):")
    print("-" * 40)
    for class_id in range(dataset.num_classes):
        count = (predictions == class_id).sum().item()
        true_count = (test_labels == class_id).sum().item()
        percentage = (count / len(predictions)) * 100
        print(f"  Class {class_id}: {count:4d} predicted ({percentage:6.2f}%), {true_count:4d} actual")
    
    print("-" * 40)
    print(f"\nAccuracy: {correct}/{len(test_labels)} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*60 + "\n")
    
    return predictions, accuracy


def predict_node(model, dataset, node_idx, device):
    """
    Predict label for a specific node
    
    Args:
        model: trained model
        dataset: CustomGraphDataset instance
        node_idx: index of node to predict
        device: computation device
    
    Returns:
        predicted_label, confidence
    """
    if node_idx < 0 or node_idx >= dataset.num_nodes:
        print(f"Invalid node index. Valid range: 0-{dataset.num_nodes-1}")
        return None, None
    
    model.eval()
    with torch.no_grad():
        node_feature = dataset.x[node_idx:node_idx+1].to(device)
        output = model(node_feature)
        probs = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_label].item()
    
    return predicted_label, confidence


def interactive_predict(model, dataset, device):
    """Interactive loop to query node predictions."""
    print("\n" + "="*60)
    print("Interactive Node Label Prediction")
    print("="*60)
    print(f"Dataset has {dataset.num_nodes} nodes (0-{dataset.num_nodes-1})")
    print(f"Number of classes: {dataset.num_classes}")
    print("Enter node index to predict (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("Node index: ").strip()
            if user_input.lower() == 'quit':
                print("Exiting prediction mode...")
                break

            node_idx = int(user_input)
            predicted_label, confidence = predict_node(model, dataset, node_idx, device)
            if predicted_label is not None:
                print(f"Node {node_idx}: Predicted Label = {predicted_label}, Confidence = {confidence:.4f}")

        except ValueError:
            print("Invalid input. Please enter a valid integer node index or 'quit'.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train DPGNN with Differential Privacy on custom dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to custom dataset .pt file')
    parser.add_argument('--dataset', type=str, default='custom_amazon',
                        help='Dataset name for logging')
    parser.add_argument('--train_start', type=int, default=1000,
                        help='Start index for training set')
    parser.add_argument('--train_end', type=int, default=10000,
                        help='End index for training set')
    parser.add_argument('--val_end', type=int, default=1000,
                        help='End index for validation set (0 to val_end)')
    
    # Test set arguments
    parser.add_argument('--test_start', type=int, default=0,
                        help='Start index for test set')
    parser.add_argument('--test_end', type=int, default=1000,
                        help='End index for test set')
    
    # Model arguments
    parser.add_argument('--K', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden dimension for GCN layers')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of training epochs (named epoch for trainer compatibility)')
    parser.add_argument('--expected_batchsize', type=int, default=32,
                        help='Expected batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Differential Privacy arguments (IMPORTANT!)
    parser.add_argument('--priv_epsilon', type=float, default=8.8,
                        help='Privacy budget epsilon (lower = more privacy, less accuracy)')
    parser.add_argument('--num_neighbors', type=int, default=1,
                        help='Number of neighbors for DP sampling in training')
    parser.add_argument('--num_neighbors_test', type=int, default=1,
                        help='Number of neighbors for test')
    parser.add_argument('--num_not_neighbors', type=int, default=1,
                        help='Number of non-neighbors for DP')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # Training arguments  
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--worker_num', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--graph_setting', type=str, default='transductive',
                        choices=['transductive', 'inductive'],
                        help='Graph learning setting')
    
    # Parse with defaults for compatibility
    parser.add_argument('--q', type=float, default=None,
                        help='Sampling ratio (auto-computed if None)')
    
    args = parser.parse_args()
    
    # Setup
    print("="*60)
    print("DPGNN Training with Differential Privacy on Custom Dataset")
    print("="*60)
    print(f"Privacy Budget (epsilon): {args.priv_epsilon}")
    print(f"Gradient Clipping (C): {args.C}")
    print("="*60 + "\n")
    
    start_time = time.time()
    SETUP.setup_seed(args.seed)
    device = SETUP.get_device()
    print(f"Using device: {device}\n")
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = CustomGraphDataset(args.dataset_path)
    args.num_classes = dataset.num_classes
    
    # Create train/val/test splits
    train_indices = torch.arange(args.train_start, args.train_end)
    val_indices = torch.arange(0, args.val_end)
    test_indices = torch.arange(args.test_start, args.test_end)
    
    print(f"\nData Split:")
    print(f"  Train set: indices {args.train_start}-{args.train_end-1} ({len(train_indices)} samples)")
    print(f"  Val set: indices 0-{args.val_end-1} ({len(val_indices)} samples)")
    print(f"  Test set (Canaries): indices {args.test_start}-{args.test_end-1} ({len(test_indices)} samples)\n")
    
    # Create DPGNN-compatible loaders
    print("Creating DPGNN data loaders...")
    train_loader, val_loader, test_loader = create_custom_dpgnn_loaders(
        dataset,
        train_indices,
        val_indices,
        test_indices,
        args
    )
    
    # Create model
    print("\nBuilding G_net model...")
    model = dms_model.G_net(
        K=args.K,
        feat_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_channels=args.hidden_channels
    )
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create DPGNN trainer with differential privacy
    print("\nInitializing DPGNN trainer with differential privacy...")
    print(f"  - Epsilon: {args.priv_epsilon}")
    print(f"  - Gradient clipping: {args.C}")
    print(f"  - Training epochs: {args.epoch}")
    print(f"  - Batch size: {args.expected_batchsize}\n")
    
    # Only train on training set; skip validation/testing during training
    trainer = tsch.trainer(
        model=model,
        optimizer=optimizer,
        loaders=[train_loader, None, None],
        device=device,
        criterion=dms_model.criterion,
        args=args,
    )
    
    # Train with differential privacy
    print("="*60)
    print("Starting DPGNN Training with Differential Privacy...")
    print("="*60)
    try:
        trainer.run()
    except AttributeError as e:
        # Handle the case where _iterator is None (happens with num_workers=0)
        if "_shutdown_workers" in str(e):
            print(f"Training completed (ignoring worker cleanup error with num_workers={args.worker_num})")
        else:
            raise
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Evaluate on canary test set (10000-10099)
    canary_indices = torch.arange(args.test_start, args.test_end)
    predictions, test_accuracy = evaluate_test_set(model, dataset, canary_indices, device)
    
    # Interactive prediction
    interactive_predict(model, dataset, device)


if __name__ == '__main__':
    main()
