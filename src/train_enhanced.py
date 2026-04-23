"""Enhanced training pipeline for hole detection using holeTrain data."""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from .parser import StepTextParser
from models.pointnet import PointNet

logger = logging.getLogger(__name__)


class AugmentedHoleDataset(Dataset):
    """Generate augmented point clouds from holeTrain STEP files."""
    
    def __init__(self, step_files: List[str], num_samples: int = 20, augmentation_factor: int = 10):
        """
        Initialize dataset with augmentation.
        
        Args:
            step_files: List of STEP file paths
            num_samples: Base number of point cloud samples per hole
            augmentation_factor: Multiplier for data augmentation
        """
        self.step_files = step_files
        self.num_samples = num_samples
        self.augmentation_factor = augmentation_factor
        self.data = []
        self.labels = []
        self._generate_augmented_data()
    
    def _generate_augmented_data(self) -> None:
        """Generate augmented training data from holeTrain files."""
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
                    # Generate multiple augmented samples per hole
                    for aug_idx in range(self.augmentation_factor):
                        for sample_idx in range(self.num_samples):
                            # Generate augmented point cloud
                            point_cloud = self._generate_augmented_hole_point_cloud(hole, aug_idx)
                            if point_cloud is not None:
                                self.data.append(point_cloud)
                                self.labels.append(feature_to_label['hole'])
                
                logger.info(f"Generated {len(parser.features.holes) * self.augmentation_factor * self.num_samples} augmented samples from {step_file}")
            except Exception as e:
                logger.warning(f"Could not parse {step_file}: {e}")
    
    def _generate_augmented_hole_point_cloud(self, hole: dict, aug_idx: int, num_points: int = 1024) -> np.ndarray:
        """Generate augmented point cloud with variations."""
        radius = hole.get('radius', 5.0)
        
        # Apply radius augmentation (±20%)
        radius_variation = 1.0 + np.random.uniform(-0.2, 0.2)
        augmented_radius = radius * radius_variation
        
        # Apply depth variation (5-15mm)
        depth = np.random.uniform(5, 15)
        
        # Apply position noise (center offset)
        center_offset = np.random.uniform(-2, 2, 2)  # x, y offset
        
        # Generate cylindrical surface points (70%)
        num_cyl_points = int(num_points * 0.7)
        theta = np.random.uniform(0, 2*np.pi, num_cyl_points)
        z = np.random.uniform(0, depth, num_cyl_points)
        
        # Add some angular noise
        theta_noise = np.random.normal(0, 0.05, num_cyl_points)
        theta = theta + theta_noise
        
        x = augmented_radius * np.cos(theta) + center_offset[0]
        y = augmented_radius * np.sin(theta) + center_offset[1]
        
        cyl_points = np.stack([x, y, z], axis=1)
        
        # Generate noise points inside hole (30%)
        noise_scale = augmented_radius + np.random.uniform(1, 3)
        noise_points = np.random.uniform(-noise_scale, noise_scale, (num_points - num_cyl_points, 3))
        
        # Add some structured noise (slight jitter to all points)
        jitter = np.random.normal(0, 0.1, (num_cyl_points, 3))
        cyl_points = cyl_points + jitter
        
        # Combine points
        point_cloud = np.vstack([cyl_points, noise_points])[:num_points]
        
        # Normalize
        point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-6)
        
        return point_cloud.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        point_cloud = torch.from_numpy(self.data[idx]).T  # (3, num_points)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return point_cloud, label


def train_enhanced_pointnet(step_files: List[str], epochs: int = 20, batch_size: int = 32, 
                          lr: float = 0.001, augmentation_factor: int = 10):
    """Train PointNet model with augmented holeTrain data."""
    
    logger.info(f"Generating augmented training dataset from {len(step_files)} holeTrain files...")
    logger.info(f"Augmentation factor: {augmentation_factor}x")
    
    dataset = AugmentedHoleDataset(step_files, num_samples=20, augmentation_factor=augmentation_factor)
    
    if len(dataset) == 0:
        logger.error("No training data generated. Ensure STEP files are valid.")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=3)
    
    logger.info(f"Training on {len(dataset)} augmented samples using device: {device}")
    logger.info(f"Dataset size: {len(dataset)} samples, batch size: {batch_size}")
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
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
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    logger.info("Training complete!")
    return model


def save_trained_model(model: nn.Module, path: str) -> None:
    """Save trained model to disk."""
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Find holeTrain STEP files
    hole_train_dir = Path("nist_sfa/holeTrain")
    step_files = sorted(list(hole_train_dir.glob("*.step")))
    
    if step_files:
        logger.info(f"Found {len(step_files)} STEP files in holeTrain")
        
        # Train with augmentation
        augmentation_factor = 15  # Generate 15x augmented samples per hole
        model = train_enhanced_pointnet(
            [str(f) for f in step_files], 
            epochs=20,
            batch_size=16,
            augmentation_factor=augmentation_factor
        )
        
        if model:
            save_trained_model(model, "models/pointnet_trained.pth")
    else:
        logger.error("No STEP files found in nist_sfa/holeTrain")
