"""MachinaQ — PointNet dataset generation and training for machining feature detection."""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .parser import StepTextParser

logger = logging.getLogger(__name__)


class PointCloudDataset(Dataset):
    """Generate point clouds from STEP geometry for training."""
    
    def __init__(self, step_files: List[str], num_points: int = 1024, num_samples: int = 5):
        self.step_files = step_files
        self.num_points = num_points
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        self._generate_training_data()
    
    def _generate_training_data(self) -> None:
        """Parse STEP files and generate point cloud samples."""
        feature_to_label = {
            'hole': 0,
            'boss': 1,
            'slot': 2,
            'thread': 3,
            'drill': 4
        }
        
        for step_file in self.step_files:
            try:
                parser = StepTextParser()
                parser.parse_file(step_file)
                
                # Extract features from parsed STEP file
                for hole in parser.features.holes:
                    for _ in range(self.num_samples):
                        # Generate synthetic point cloud around hole
                        point_cloud = self._generate_hole_point_cloud(hole)
                        if point_cloud is not None:
                            self.data.append(point_cloud)
                            self.labels.append(feature_to_label['hole'])
                
                logger.info(f"Generated {len(parser.features.holes)} hole samples from {step_file}")
            except Exception as e:
                logger.warning(f"Could not parse {step_file}: {e}")
    
    def _generate_hole_point_cloud(self, hole: dict) -> np.ndarray:
        """Generate synthetic point cloud for a hole feature."""
        radius = hole.get('radius', 5.0)
        
        # Generate points on cylindrical surface
        num_points_cyl = int(self.num_points * 0.7)
        theta = np.random.uniform(0, 2*np.pi, num_points_cyl)
        z = np.random.uniform(0, 10, num_points_cyl)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        cyl_points = np.stack([x, y, z], axis=1)
        
        # Generate points inside hole (noise)
        noise_points = np.random.uniform(-radius-1, radius+1, (self.num_points - num_points_cyl, 3))
        
        # Combine points
        point_cloud = np.vstack([cyl_points, noise_points])[:self.num_points]
        
        # Normalize
        point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-6)
        
        return point_cloud.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        point_cloud = torch.tensor(self.data[idx].tolist(), dtype=torch.float32).T  # (3, num_points)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return point_cloud, label


def train_pointnet(step_files: List[str], epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    """Train PointNet model on STEP features."""
    # Import inside function to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.pointnet import PointNet
    
    logger.info("Generating training dataset from NIST STEP files...")
    dataset = PointCloudDataset(step_files, num_samples=3)
    
    if len(dataset) == 0:
        logger.error("No training data generated. Ensure STEP files are valid.")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Training on {len(dataset)} samples using device: {device}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (point_clouds, labels) in enumerate(dataloader):
            point_clouds = point_clouds.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(point_clouds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    logger.info("Training complete!")
    return model


def save_trained_model(model: nn.Module, path: str) -> None:
    """Save trained model to disk."""
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Find NIST STEP files
    nist_dir = Path("nist_sfa")
    step_files = sorted(list(nist_dir.glob("nist_*.stp")))  # Use all available files
    
    if step_files:
        logger.info(f"Found {len(step_files)} NIST STEP files for training")
        model = train_pointnet([str(f) for f in step_files], epochs=10)
        if model:
            save_trained_model(model, "models/machinaq_pointnet.pth")
    else:
        print("No NIST STEP files found in nist_sfa directory")