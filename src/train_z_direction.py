"""Train PointNet model on NIST data for holes along Z direction."""

import logging
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.parser import StepTextParser
from models.pointnet import PointNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class ZDirectionHoleDataset(Dataset):
    """Dataset of point clouds for Z-direction aligned holes from NIST files."""
    
    def __init__(self, step_files: list[str], z_tolerance: float = 0.1):
        """
        Initialize dataset with Z-direction hole filtering.
        
        Args:
            step_files: List of STEP file paths
            z_tolerance: Tolerance for Z-direction alignment (cosine similarity threshold)
        """
        self.samples = []
        self.z_tolerance = z_tolerance  # Cosine similarity threshold for Z alignment
        
        # Parse NIST files and extract Z-direction holes
        for step_file in step_files:
            self._load_z_direction_holes(step_file)
    
    def _load_z_direction_holes(self, step_file: str) -> None:
        """Extract holes aligned along Z direction from STEP file."""
        try:
            parser = StepTextParser()
            parser.parse_file(step_file)
            
            features = parser.features
            if not features.holes:
                return
            
            for hole in features.holes:
                # Check if hole is aligned along Z direction
                # Z-direction means axis direction is close to (0, 0, ±1)
                if self._is_z_aligned(hole):
                    # Generate point cloud from hole geometry
                    point_cloud = self._generate_point_cloud_from_hole(hole)
                    if point_cloud is not None and len(point_cloud) >= 1024:
                        # Normalize to 1024 points
                        indices = np.random.choice(len(point_cloud), 1024, replace=len(point_cloud) < 1024)
                        point_cloud_normalized = point_cloud[indices]
                        
                        self.samples.append({
                            'point_cloud': point_cloud_normalized,
                            'label': 0,  # Class 0: hole
                            'radius': hole.get('radius', 0),
                            'file': step_file
                        })
                        
            logger.info(f"Generated {len([s for s in self.samples if s['file'] == step_file])} "
                       f"Z-direction hole samples from {step_file}")
                       
        except Exception as e:
            logger.warning(f"Error processing {step_file}: {e}")
    
    def _is_z_aligned(self, hole: dict) -> bool:
        """Check if hole is aligned along Z direction (axis ~ [0, 0, ±1]).
        
        For now, we assume all detected holes are Z-direction aligned,
        as most manufacturing features in NIST files are vertical (Z-direction).
        This can be enhanced with actual axis extraction from STEP files.
        """
        # All holes default to Z-direction for this training
        return True
    
    def _generate_point_cloud_from_hole(self, hole: dict) -> np.ndarray:
        """Generate synthetic point cloud from hole geometry."""
        radius = hole.get('radius', 5.0)
        depth = hole.get('depth', 10.0)
        
        if radius <= 0 or depth <= 0:
            return None
        
        # Generate points on cylindrical surface (70%) + noise (30%)
        n_surface = int(1024 * 0.7)
        n_noise = 1024 - n_surface
        
        points = []
        
        # Points on cylindrical surface
        theta = np.random.uniform(0, 2 * np.pi, n_surface)
        z = np.random.uniform(-depth / 2, depth / 2, n_surface)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        surface_points = np.column_stack([x, y, z])
        points.append(surface_points)
        
        # Noise points (random points in bounding box)
        noise_x = np.random.uniform(-radius * 1.5, radius * 1.5, n_noise)
        noise_y = np.random.uniform(-radius * 1.5, radius * 1.5, n_noise)
        noise_z = np.random.uniform(-depth / 2, depth / 2, n_noise)
        noise_points = np.column_stack([noise_x, noise_y, noise_z])
        points.append(noise_points)
        
        point_cloud = np.vstack(points)
        
        # Normalize via z-score
        mean = point_cloud.mean(axis=0)
        std = point_cloud.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        point_cloud = (point_cloud - mean) / std
        
        return point_cloud
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]
        point_cloud = torch.FloatTensor(sample['point_cloud']).T  # (3, 1024)
        label = torch.LongTensor([sample['label']])
        return point_cloud, label.squeeze()


def train_model(dataset: Dataset, epochs: int = 20, batch_size: int = 16) -> None:
    """Train PointNet on Z-direction hole dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Dataset size: {len(dataset)} samples, batch size: {batch_size}")
    
    # Initialize model
    model = PointNet(num_classes=5)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    best_acc = 0
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for point_clouds, labels in dataloader:
            point_clouds = point_clouds.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(point_clouds)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Calculate epoch accuracy
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/pointnet_trained.pth')
            logger.info(f"  → New best accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                break
    
    logger.info("Training complete!")
    logger.info(f"Model saved to models/pointnet_trained.pth")


def main():
    """Main training function."""
    # Find NIST STEP files
    nist_dir = 'nist_sfa'
    step_files = [
        os.path.join(nist_dir, f) for f in os.listdir(nist_dir)
        if f.endswith('.stp')
    ]
    
    if not step_files:
        logger.error(f"No STEP files found in {nist_dir}")
        return
    
    logger.info(f"Found {len(step_files)} NIST STEP files")
    
    # Create dataset (Z-direction holes only)
    logger.info("Extracting Z-direction aligned holes from NIST files...")
    dataset = ZDirectionHoleDataset(step_files, z_tolerance=0.85)
    
    if len(dataset) < 10:
        logger.warning(f"Dataset too small: {len(dataset)} samples. "
                      "Consider lowering z_tolerance for more samples.")
    
    logger.info(f"Training dataset ready: {len(dataset)} Z-direction hole samples")
    
    if len(dataset) == 0:
        logger.error("No Z-direction holes found in NIST files!")
        return
    
    # Train model
    train_model(dataset, epochs=20, batch_size=16)


if __name__ == '__main__':
    main()
