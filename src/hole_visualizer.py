"""Extract and visualize holes from STEP PCURVE and surface definitions."""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .parser import StepTextParser


class PCurveExtractor:
    """Extract PCURVE entities and their parametric representations from STEP files."""

    def __init__(self, step_file: str):
        self.step_file = step_file
        self.parser = StepTextParser()
        self.parser.parse_file(step_file)
        self.entities = self._parse_all_entities()
        self.pcurves = self._extract_pcurves()
        self.surfaces = self._extract_surfaces()

    def _parse_all_entities(self) -> Dict[int, Dict]:
        """Parse all entities from STEP file into a dictionary."""
        with open(self.step_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        entities = {}
        # Match entity patterns: #ID = ENTITY_TYPE(...)
        pattern = r'#(\d+)\s*=\s*([A-Z_]+)\s*\((.*?(?:[,;]))\s*\);'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            entity_id = int(match.group(1))
            entity_type = match.group(2)
            entity_data = match.group(3)
            
            entities[entity_id] = {
                'type': entity_type,
                'data': entity_data.strip(),
                'raw': match.group(0)
            }
        
        return entities

    def _extract_pcurves(self) -> Dict[int, Dict]:
        """Extract all PCURVE entities."""
        pcurves = {}
        
        for entity_id, entity in self.entities.items():
            if entity['type'] == 'PCURVE':
                pcurves[entity_id] = self._parse_pcurve(entity_id, entity)
        
        return pcurves

    def _extract_surfaces(self) -> Dict[int, Dict]:
        """Extract surface definitions (PLANE, CYLINDRICAL_SURFACE, etc.)."""
        surfaces = {}
        
        surface_types = ['PLANE', 'CYLINDRICAL_SURFACE', 'TOROIDAL_SURFACE', 
                         'SPHERICAL_SURFACE', 'SURFACE_OF_REVOLUTION']
        
        for entity_id, entity in self.entities.items():
            if entity['type'] in surface_types:
                surfaces[entity_id] = self._parse_surface(entity_id, entity)
        
        return surfaces

    def _parse_pcurve(self, entity_id: int, entity: Dict) -> Dict:
        """Parse a PCURVE entity."""
        # PCURVE('', surface_ref, parametric_rep)
        data = entity['data']
        
        # Extract surface reference and parametric representation reference
        parts = re.findall(r'#(\d+)', data)
        if len(parts) >= 2:
            surface_id = int(parts[0])
            param_rep_id = int(parts[1])
            
            return {
                'surface_id': surface_id,
                'param_rep_id': param_rep_id,
                'surface': self.entities.get(surface_id),
                'param_rep': self.entities.get(param_rep_id)
            }
        
        return {}

    def _parse_surface(self, entity_id: int, entity: Dict) -> Dict:
        """Parse surface entity."""
        entity_type = entity['type']
        data = entity['data']
        
        if entity_type == 'PLANE':
            # Extract placement reference (AXIS2_PLACEMENT_3D)
            placement_match = re.search(r'#(\d+)', data)
            if placement_match:
                placement_id = int(placement_match.group(1))
                return {
                    'type': 'PLANE',
                    'placement_id': placement_id,
                    'placement': self._parse_axis2_placement(placement_id)
                }
        
        elif entity_type == 'CYLINDRICAL_SURFACE':
            # CYLINDRICAL_SURFACE('', placement, radius)
            matches = re.findall(r'#(\d+)|([0-9.]+)', data)
            if matches:
                placement_id = int(matches[0][0]) if matches[0][0] else None
                radius = float(matches[1][1]) if len(matches) > 1 and matches[1][1] else 5.0
                
                return {
                    'type': 'CYLINDRICAL_SURFACE',
                    'placement_id': placement_id,
                    'radius': radius,
                    'placement': self._parse_axis2_placement(placement_id) if placement_id else None
                }
        
        return {'type': entity_type}

    def _parse_axis2_placement(self, entity_id: int) -> Optional[Dict]:
        """Parse AXIS2_PLACEMENT_3D entity."""
        if entity_id not in self.entities:
            return None
        
        entity = self.entities[entity_id]
        if entity['type'] != 'AXIS2_PLACEMENT_3D':
            return None
        
        data = entity['data']
        
        # Extract origin, direction, ref_direction references
        point_ids = re.findall(r'#(\d+)', data)
        
        placement = {'origin': None, 'z_axis': None, 'x_axis': None}
        
        if len(point_ids) >= 1:
            origin_id = int(point_ids[0])
            placement['origin'] = self._parse_cartesian_point(origin_id)
        
        if len(point_ids) >= 2:
            z_axis_id = int(point_ids[1])
            placement['z_axis'] = self._parse_direction(z_axis_id)
        else:
            placement['z_axis'] = np.array([0, 0, 1])
        
        if len(point_ids) >= 3:
            x_axis_id = int(point_ids[2])
            placement['x_axis'] = self._parse_direction(x_axis_id)
        else:
            placement['x_axis'] = np.array([1, 0, 0])
        
        return placement

    def _parse_cartesian_point(self, entity_id: int) -> Optional[np.ndarray]:
        """Parse CARTESIAN_POINT entity."""
        if entity_id not in self.entities:
            return None
        
        entity = self.entities[entity_id]
        if entity['type'] != 'CARTESIAN_POINT':
            return None
        
        data = entity['data']
        # Extract coordinates: (x, y, z)
        coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', data)
        
        if len(coords) >= 3:
            return np.array([float(coords[0]), float(coords[1]), float(coords[2])])
        elif len(coords) >= 2:
            return np.array([float(coords[0]), float(coords[1])])
        
        return None

    def _parse_direction(self, entity_id: int) -> Optional[np.ndarray]:
        """Parse DIRECTION entity."""
        if entity_id not in self.entities:
            return None
        
        entity = self.entities[entity_id]
        if entity['type'] != 'DIRECTION':
            return None
        
        data = entity['data']
        # Extract direction components
        coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', data)
        
        if len(coords) >= 3:
            direction = np.array([float(coords[0]), float(coords[1]), float(coords[2])])
            norm = np.linalg.norm(direction)
            if norm > 0:
                return direction / norm
            return direction
        elif len(coords) >= 2:
            direction = np.array([float(coords[0]), float(coords[1])])
            norm = np.linalg.norm(direction)
            if norm > 0:
                return direction / norm
            return direction
        
        return None

    def get_hole_features(self) -> List[Dict]:
        """Extract hole features from parsed entities."""
        holes = []
        
        # Use parser's detected holes
        for hole in self.parser.features.holes:
            holes.append({
                'type': 'hole',
                'radius': hole.get('radius', 5.0),
                'center': hole.get('center'),
                'id': hole.get('id')
            })
        
        return holes

    def get_pcurve_coordinates(self, pcurve_id: int, num_points: int = 100) -> Optional[np.ndarray]:
        """Evaluate a PCURVE in 3D space."""
        if pcurve_id not in self.pcurves:
            return None
        
        pcurve = self.pcurves[pcurve_id]
        surface_id = pcurve.get('surface_id')
        
        if surface_id not in self.surfaces:
            return None
        
        surface = self.surfaces[surface_id]
        
        # For now, return parametric curve points for plotting
        # In a full implementation, we'd evaluate the parametric curve
        return None


class HoleVisualizer:
    """Visualize hole features from STEP files."""

    @staticmethod
    def plot_hole_from_step(step_file: str, output_path: Optional[str] = None):
        """Extract and plot hole from STEP file."""
        extractor = PCurveExtractor(step_file)
        holes = extractor.get_hole_features()
        
        if not holes:
            print(f"No holes found in {step_file}")
            return
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D view
        ax3d = fig.add_subplot(121, projection='3d')
        
        # Extract hole geometry
        for idx, hole in enumerate(holes):
            radius = hole.get('radius', 5.0)
            center = hole.get('center')
            
            # Use default center if not available
            if center is None:
                center = [0, 0, 5]
            elif not isinstance(center, (list, tuple, np.ndarray)):
                center = [0, 0, 5]
            else:
                center = list(center) if isinstance(center, np.ndarray) else center
                if len(center) < 3:
                    center = center + [5] * (3 - len(center))
            
            # Generate cylindrical hole surface
            theta = np.linspace(0, 2*np.pi, 100)
            z = np.linspace(0, 10, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            
            x_grid = radius * np.cos(theta_grid) + center[0]
            y_grid = radius * np.sin(theta_grid) + center[1]
            z_grid = z_grid + center[2]
            
            # Plot hole surface
            ax3d.plot_surface(x_grid, y_grid, z_grid, alpha=0.7, cmap='viridis')
        
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Z (mm)')
        ax3d.set_title('3D Hole Visualization')
        
        # 2D view (top-down)
        ax2d = fig.add_subplot(122)
        
        for hole in holes:
            radius = hole.get('radius', 5.0)
            center = hole.get('center')
            
            # Use default center if not available
            if center is None:
                center = [0, 0, 5]
            elif not isinstance(center, (list, tuple, np.ndarray)):
                center = [0, 0, 5]
            else:
                center = list(center) if isinstance(center, np.ndarray) else center
                if len(center) < 2:
                    center = center + [0] * (2 - len(center))
            
            circle = plt.Circle((center[0], center[1]), radius, 
                              fill=False, edgecolor='b', linewidth=2)
            ax2d.add_patch(circle)
            
            ax2d.plot(center[0], center[1], 'r+', markersize=10, label='Hole center')
        
        ax2d.set_xlim(-30, 30)
        ax2d.set_ylim(-30, 30)
        ax2d.set_aspect('equal')
        ax2d.grid(True, alpha=0.3)
        ax2d.set_xlabel('X (mm)')
        ax2d.set_ylabel('Y (mm)')
        ax2d.set_title('Top View (2D Projection)')
        ax2d.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

    @staticmethod
    def plot_multiple_holes(step_files: List[str], output_dir: Optional[str] = None):
        """Plot holes from multiple STEP files."""
        for step_file in step_files:
            filename = Path(step_file).stem
            output_path = None
            
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_path = output_dir_path / f"{filename}_hole_visualization.png"
            
            print(f"Processing {filename}...")
            HoleVisualizer.plot_hole_from_step(step_file, str(output_path) if output_path else None)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with holeTrain files
    hole_train_dir = Path("nist_sfa/holeTrain")
    step_files = sorted(list(hole_train_dir.glob("*.step")))
    
    if step_files:
        print(f"Found {len(step_files)} STEP files for visualization")
        output_dir = "outputs/hole_visualizations"
        HoleVisualizer.plot_multiple_holes([str(f) for f in step_files], output_dir)
    else:
        print("No STEP files found in nist_sfa/holeTrain")
