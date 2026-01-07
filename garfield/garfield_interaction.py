"""Helper functions for interacting/visualization with GARField model."""
from typing import List, Optional, Tuple, Union
import viser
import trimesh
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared

from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

from garfield.garfield_model import GarfieldModel

class GarfieldClickScene(nn.Module):
    """UI for clicking on a scene (visualized as spheres).
    This needs to be a nn.Module to allow the viewer to register callbacks.
    """
    _click_handle: viser.GlbHandle
    _box_handle: viser.GlbHandle
    selected_location: np.ndarray
    scale_handle: ViewerSlider  # For getting the scale to query GARField
    model_handle: List[GarfieldModel]  # Store as list to avoid circular children

    def __init__(
            self,
            device: torch.device,
            scale_handle: ViewerSlider,
            model_handle: List[GarfieldModel]
        ):
        super().__init__()
        self.add_click_button: ViewerButton = ViewerButton(
            name="Click", cb_hook=self._add_click_cb
        )
        self.del_click_button: ViewerButton = ViewerButton(
            name="Reset Click", cb_hook=self._del_click_cb
        )
        self.viewer_control: ViewerControl = ViewerControl()

        self.scale_handle = scale_handle
        self.model_handle = model_handle
        self.scale_handle.cb_hook = self._update_scale_vis

        self._click_handle = None
        self._box_handle = None
        self._center_handle = None  # Handle for object center visualization
        self._selected_points_handles = []  # Handles for selected near points visualization
        self.selected_location = None
        self.device = device
        ## changed
        self.save_dir = Path("outputs/clicked_pointclouds")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.click_count = 0
        # For cumulative average of object center
        self._accumulated_center = None  # Accumulated center sum
        self._center_count = 0  # Number of centers accumulated

    def _add_click_cb(self, button: ViewerButton):
        """Button press registers a click event, which will add a sphere.
        Refer more to nerfstudio docs for more details. """
        self.add_click_button.set_disabled(True)
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.add_click_button.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Also keep track of the selected location."""

        origin = torch.tensor(click.origin).view(1, 3)
        direction = torch.tensor(click.direction).view(1, 3)

        # get intersection
        bundle = RayBundle(
            origin,
            direction,
            torch.tensor(0.001).view(1, 1),
            nears=torch.tensor(0.05).view(1, 1),
            fars=torch.tensor(100).view(1, 1),
            camera_indices=torch.tensor(0).view(1, 1),
        ).to(self.device)

        # Get the distance/depth to the intersection --> calculate 3D position of the click
        model = self.model_handle[0]
        ray_samples, _, _ = model.proposal_sampler(bundle, density_fns=model.density_fns)
        field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
        if model.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        with torch.no_grad():
            depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)
        distance = depth[0, 0].detach().cpu().numpy()
        click_position = np.array(origin + direction * distance) * VISER_NERFSTUDIO_SCALE_RATIO
        print(f"Clicked position (3D world coords): {click_position}, distance: {distance}")

        # Update click visualization
        self._del_click_cb(None)
        sphere_mesh: trimesh.Trimesh = trimesh.creation.icosphere(radius=0.1)
        sphere_mesh.vertices += click_position
        sphere_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)  # type: ignore
        sphere_mesh_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/hit_pos", mesh=sphere_mesh
        )
        self._click_handle = sphere_mesh_handle
        self.selected_location = np.array(origin + direction * distance)
        self._update_scale_vis(self.scale_handle)

    def _del_click_cb(self, button: ViewerButton):
        """Remove the click location and click visualizations."""
        if self._click_handle is not None:
            self._click_handle.remove()
        self._click_handle = None
        if self._box_handle is not None:
            self._box_handle.remove()
        self._box_handle = None
        if self._center_handle is not None:
            self._center_handle.remove()
        self._center_handle = None
        # Remove all selected points visualizations
        for handle in self._selected_points_handles:
            if handle is not None:
                handle.remove()
        self._selected_points_handles = []
        self.selected_location = None
        # Reset accumulated center when click is reset
        self._accumulated_center = None
        self._center_count = 0

    def _update_scale_vis(self, slider: ViewerSlider):
        """Update the scale visualization."""
        if self._box_handle is not None:
            self._box_handle.remove()
            self._box_handle = None
        if self.selected_location is not None:
            box_mesh = trimesh.creation.icosphere(radius=VISER_NERFSTUDIO_SCALE_RATIO*max(0.001, slider.value)/2, subdivision=0)
            self._box_handle = self.viewer_control.viser_server.add_mesh_simple(
                name=f"/hit_pos_box", 
                vertices=box_mesh.vertices,
                faces=box_mesh.faces,
                position=(self.selected_location * VISER_NERFSTUDIO_SCALE_RATIO).flatten(),
                wireframe=True
            )
        # Reset accumulated center when scale value changes
        self._accumulated_center = None
        self._center_count = 0

    def get_outputs(self, outputs: dict):
        """Visualize affinity between the selected 3D point and the points visibl in current rendered view."""
        if self.selected_location is None:
            return None

        location = self.selected_location
        instance_scale = self.scale_handle.value
        print(f"input location is : {location}")
        
        # Initialize object_center
        object_center = torch.zeros((1, 3), device=self.device)
        
        # mimic the fields call
        grouping_field = self.model_handle[0].grouping_field
        positions = torch.tensor(location).view(1, 3).to(self.device)
        positions = grouping_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        xs = [e(positions.view(-1, 3)) for e in grouping_field.enc_list]
        x = torch.concat(xs, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)
        instance_pass = grouping_field.get_mlp(x, torch.tensor([instance_scale]).to(self.device).view(1, 1)) # [1, 256]

        # Extract point cloud and calculate object center if ray sample positions are available
        if "ray_sample_positions" in outputs and "ray_sample_weights" in outputs:
            ray_positions = outputs["ray_sample_positions"]  # [num_rays, num_samples, 3]
            ray_weights = outputs["ray_sample_weights"]  # [num_rays, num_samples]
            
            # Flatten to get all 3D points
            all_points = ray_positions.view(-1, 3)  # [num_rays * num_samples, 3]
            # print(f"all_points.shape: {all_points.shape}, ray_positions.shape: {ray_positions.shape}")
            all_weights = ray_weights.view(-1)  # [num_rays * num_samples]
            
            # Filter points by distance from clicked location
            click_location_tensor = torch.tensor(location, device=self.device).view(1, 3)
            distances = torch.norm(all_points - click_location_tensor, p=2, dim=-1)  # [num_rays * num_samples]
            
            # 1. Filter points within instance_scale distance and with significant weight
            weight_threshold = 0.001  # Only consider points with meaningful density
            # distance_mask = distances <= instance_scale
            distance_mask = distances <= VISER_NERFSTUDIO_SCALE_RATIO*max(0.001, instance_scale)/2
            # weight_mask = all_weights > weight_threshold
            # valid_mask = distance_mask & weight_mask
            near_points = all_points[distance_mask]  # [N, 3]            
            
            if len(near_points) > 0:
                # print(f"===near_points.shape: {near_points.shape}===")
                # 2. Get instance values for each near_point using get_mlp
                # Preprocess near_points: spatial distortion and normalization
                near_positions = near_points.to(self.device)  # [N, 3]
                near_positions = grouping_field.spatial_distortion(near_positions)
                near_positions = (near_positions + 2.0) / 4.0
                
                # Compute hash encoding for all near_points
                xs = [e(near_positions.view(-1, 3)) for e in grouping_field.enc_list]
                x = torch.concat(xs, dim=-1)  # [N, tot_out_dims]
                x = x / x.norm(dim=-1, keepdim=True)  # Normalize hash encoding
                
                # Get instance values for all near_points
                instance_scales_near = torch.ones(near_points.shape[0], 1, device=self.device) * instance_scale
                near_instance_values = grouping_field.get_mlp(x, instance_scales_near)  # [N, n_instance_dims]
                
                # 3. Calculate interact values between position and near_points, normalize and filter
                # instance_pass: [1, n_instance_dims], near_instance_values: [N, n_instance_dims]
                interact_distances = torch.norm(near_instance_values - instance_pass.float(), p=2, dim=-1)  # [N]
                
                # Normalize interact distances (assuming max possible distance is sqrt(2) for unit vectors)
                # For normalized vectors, max L2 distance is 2.0, so we normalize by dividing by 2.0
                # max_possible_distance = 2.0
                # normalized_interact = interact_distances / max_possible_distance  # [N]
                
                # Filter points with normalized interact <= 1.0 (all points since normalized, but we keep the threshold)
                # Actually, since we're normalizing by max distance, values should be in [0, 1]
                # But to be safe, we use a threshold (e.g., 0.5 or keep all)
                interact_mask = interact_distances <= 0.2
                # print(f"interact_mask.shape: {interact_mask.shape}")
                # selected_near_points = near_positions[interact_mask]  # [M, 3] where M <= N 
                selected_near_points = near_points[interact_mask]

                
                # Visualize all selected near points
                # self._visualize_selected_points(selected_near_points)
                
                if len(selected_near_points) > 0:
                    # 4. Calculate center from selected near_points
                    center = torch.mean(selected_near_points, dim=0)  # [3]
                    current_center = center.cpu().numpy()  # [3]                    
                    # Update cumulative average
                    if self._accumulated_center is None:
                        self._accumulated_center = current_center.copy()
                        self._center_count = 1
                    else:
                        # Incremental average: new_avg = (old_avg * count + new_value) / (count + 1)
                        self._accumulated_center = (self._accumulated_center * self._center_count + current_center) / (self._center_count + 1)
                        self._center_count += 1
                    
                    # Visualize cumulative average center as a green sphere
                    if self._accumulated_center is not None:
                        self._visualize_center(self._accumulated_center)
                        print(f"cumulative_avg_center (count={self._center_count}): {self._accumulated_center}")
                        
                        # Apply inverse normalization process to accumulated_center
                        # Reverse process: (near_positions + 2.0) / 4.0 -> near_positions * 4.0 - 2.0
                        # Then reverse spatial_distortion
                        # accumulated_center_tensor = torch.tensor(self._accumulated_center, device=self.device).view(1, 3)
                        
                        # # Step 1: Reverse normalization: (x + 2.0) / 4.0 -> x * 4.0 - 2.0
                        # denormalized_center = accumulated_center_tensor * 4.0 - 2.0
                        
                        # # Step 2: Reverse spatial_distortion (SceneContraction inverse)
                        # # SceneContraction: y = (2 - 1/||x||) * (x/||x||) for ||x|| > 1, y = x for ||x|| <= 1
                        # # Inverse: x = y for ||y|| <= 1, x = (1/(2 - ||y||)) * (y/||y||) for 1 < ||y|| < 2
                        # def uncontract(positions):
                        #     mag = torch.linalg.norm(positions, dim=-1, keepdim=True)  # [N, 1]
                        #     # For ||y|| <= 1, no contraction was applied, so x = y
                        #     # For 1 < ||y|| < 2, we need to uncontract
                        #     # For ||y|| >= 2, contraction is at maximum (edge case)
                        #     mask_contracted = (mag > 1.0).squeeze(-1)  # [N]
                        #     uncontracted = positions.clone()
                        #     if mask_contracted.any():
                        #         # Apply uncontract only to contracted positions
                        #         mag_contracted = mag[mask_contracted]  # [M, 1]
                        #         pos_contracted = positions[mask_contracted]  # [M, 3]
                        #         uncontracted[mask_contracted] = (1.0 / (2.0 - mag_contracted)) * (pos_contracted / mag_contracted)
                        #     return uncontracted
                        
                        # uncontract_center = uncontract(denormalized_center)
                        # uncontract_center_cpu = uncontract_center.cpu().numpy().flatten()
                        # print(f"cumulative_avg_center (denormalized, uncontract): {uncontract_center_cpu}")
            
        return {
            "object_center": object_center,
            "instance_interact": torch.norm(outputs['instance'] - instance_pass.float(), p=2, dim=-1) if 'instance' in outputs else torch.tensor([])
        }
    
    def _visualize_center(self, center: np.ndarray):
        """Visualize the object center as a green sphere in the viewer."""
        # Remove previous center visualization if exists
        if self._center_handle is not None:
            self._center_handle.remove()
        
        # Create a green sphere at the center position
        center_position = center * VISER_NERFSTUDIO_SCALE_RATIO
        center_sphere = trimesh.creation.icosphere(radius=0.08)
        center_sphere.vertices += center_position
        center_sphere.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # Green color
        
        # Add to viewer
        self._center_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/object_center", mesh=center_sphere
        )
    
    def _visualize_selected_points(self, selected_points: torch.Tensor):
        """Visualize all selected near points as small spheres in the viewer."""
        # Remove previous selected points visualizations
        for handle in self._selected_points_handles:
            if handle is not None:
                handle.remove()
        self._selected_points_handles = []
        
        if len(selected_points) == 0:
            return
        
        # Convert to numpy and scale
        selected_points_np = selected_points.cpu().numpy()  # [M, 3]
        selected_points_scaled = selected_points_np * VISER_NERFSTUDIO_SCALE_RATIO
        
        # Create a small sphere template
        sphere_template = trimesh.creation.icosphere(radius=0.03, subdivision=0)
        
        # Visualize each point as a small blue sphere
        for i, point in enumerate(selected_points_scaled):
            sphere_mesh = sphere_template.copy()
            sphere_mesh.vertices += point
            sphere_mesh.visual.vertex_colors = (0.0, 0.0, 1.0, 1.0)  # Blue color
            
            # Add to viewer
            handle = self.viewer_control.viser_server.add_mesh_trimesh(
                name=f"/selected_point_{i}", mesh=sphere_mesh
            )
            self._selected_points_handles.append(handle)
        