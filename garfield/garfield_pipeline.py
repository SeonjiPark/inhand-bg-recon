import typing
import time
import os
import atexit
import yaml
from dataclasses import dataclass, field
from typing import Literal, Type, Mapping, Any
from pathlib import Path

import torch
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler

import tqdm

from sklearn.preprocessing import QuantileTransformer
from garfield.garfield_datamanager import GarfieldDataManagerConfig, GarfieldDataManager
from garfield.garfield_model import GarfieldModel, GarfieldModelConfig


@dataclass
class GarfieldPipelineConfig(VanillaPipelineConfig):
    """Configuration for GARField pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GarfieldPipeline)
    """target class to instantiate"""

    datamanager: GarfieldDataManagerConfig = field(default_factory=lambda: GarfieldDataManagerConfig())
    model: GarfieldModelConfig = field(default_factory=lambda: GarfieldModelConfig())

    start_grouping_step: int = 2000
    max_grouping_scale: float = 2.0
    num_rays_per_image: int = 256
    normalize_grouping_scale: bool = True


class GarfieldPipeline(VanillaPipeline):
    config: GarfieldPipelineConfig
    datamanager: GarfieldDataManager
    model: GarfieldModel

    def __init__(
        self,
        config: GarfieldPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        config.model.max_grouping_scale = config.max_grouping_scale
        super().__init__(
            config,
            device,
            test_mode,
            world_size,
            local_rank,
            grad_scaler,
        )
        
        # Initialize training metrics tracking
        if test_mode != "inference" and local_rank == 0:
            self._training_start_time = time.time()
            self._max_gpu_memory_mb = 0.0
            self._results_saved = False
            self._max_num_iterations = None  # Will be set when we know it
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Register exit handler to save results when program exits
            # This ensures results are saved even if training is interrupted
            atexit.register(self._save_results_on_exit)

    def get_train_loss_dict(self, step: int):
        """In addition to the base class, we also calculate SAM masks
        and their 3D scales at `start_grouping_step`."""

        # Track GPU memory usage from cursor
        if hasattr(self, '_training_start_time') and torch.cuda.is_available():
            current_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            self._max_gpu_memory_mb = max(self._max_gpu_memory_mb, current_memory_mb)
        
        # Try to get max_num_iterations from config if available
        # This is a workaround since we don't have direct access to TrainerConfig
        if hasattr(self, '_training_start_time') and self._max_num_iterations is None:
            # Try to find max_num_iterations from saved config.yml
            try:
                outputs_dir = Path("outputs")
                if outputs_dir.exists():
                    config_files = list(outputs_dir.rglob("config.yml"))
                    if config_files:
                        config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        with open(config_files[0], 'r') as f:
                            config = yaml.safe_load(f)
                            if 'max_num_iterations' in config:
                                self._max_num_iterations = config['max_num_iterations']
            except:
                pass
        
        # Save results when we reach the last step
        if (hasattr(self, '_training_start_time') and 
            not self._results_saved):
            # If we know max_num_iterations, check if we're at the end
            if self._max_num_iterations is not None:
                if step >= self._max_num_iterations - 1:
                    self.save_training_results(self._max_num_iterations)
                    self._results_saved = True
            # Otherwise, save periodically as backup (will be overwritten by final save)
            elif step % 1000 == 0:
                # Use current step as estimate if we don't know max_num_iterations
                self.save_training_results(step + 1)
        
        ### to cursor
        if step == self.config.start_grouping_step:
            loaded = self.datamanager.load_sam_data()
            if not loaded:
                self.populate_grouping_info()
            else:
                # Initialize grouping statistics. This will be automatically loaded from a checkpoint next time.
                scale_stats = self.datamanager.scale_3d_statistics
                self.grouping_stats = torch.nn.Parameter(scale_stats)
                self.model.grouping_field.quantile_transformer = (
                    self._get_quantile_func(scale_stats)
                )
            # Set the number of rays per image to the number of rays per image for grouping
            pixel_sampler = self.datamanager.train_pixel_sampler
            pixel_sampler.num_rays_per_image = pixel_sampler.config.num_rays_per_image

        ray_bundle, batch = self.datamanager.next_train(step)
        if step >= self.config.start_grouping_step:
            # also set the grouping info in the batch; in-place operation
            self.datamanager.next_group(ray_bundle, batch)

        model_outputs = self._model(
            ray_bundle
        )  # train distributed data parallel model if world_size > 1

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        if step >= self.config.start_grouping_step:
            loss_dict.update(
                self.model.get_loss_dict_group(model_outputs, batch, metrics_dict)
            )

        return model_outputs, loss_dict, metrics_dict

    def populate_grouping_info(self):
        """
        Calculate groups from SAM and their 3D scales, and save them in the datamanager.
        This information is required to supervise the grouping field.
        """
        # Note that pipeline is in train mode here, via the base trainer.
        self.model.eval()

        # Calculate multi-scale masks, and their 3D scales
        scales_3d_list, pixel_level_keys_list, group_cdf_list = [], [], []
        train_cameras = self.datamanager.train_dataset.cameras
        for i in tqdm.trange(len(train_cameras), desc="Calculating 3D masks"):
            camera_ray_bundle = train_cameras.generate_rays(camera_indices=i).to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle
                )

            # Get RGB (for SAM mask generation), depth and 3D point locations (for 3D scale calculation)
            rgb = self.datamanager.train_dataset[i]["image"]
            depth = outputs["depth"]
            points = camera_ray_bundle.origins + camera_ray_bundle.directions * depth
            # Scales are capped to `max_grouping_scale` to filter noisy / outlier masks.
            (
                pixel_level_keys,
                scale_3d,
                group_cdf,
            ) = self.datamanager._calculate_3d_groups(
                rgb, depth, points, max_scale=self.config.max_grouping_scale
            )

            pixel_level_keys_list.append(pixel_level_keys)
            scales_3d_list.append(scale_3d)
            group_cdf_list.append(group_cdf)

        # Save grouping data, and set it in the datamanager for current training.
        # This will be cached, so we don't need to calculate it again.
        self.datamanager.save_sam_data(
            pixel_level_keys_list, scales_3d_list, group_cdf_list
        )
        self.datamanager.pixel_level_keys = torch.nested.nested_tensor(
            pixel_level_keys_list
        )
        self.datamanager.scale_3d = torch.nested.nested_tensor(scales_3d_list)
        self.datamanager.group_cdf = torch.nested.nested_tensor(group_cdf_list)

        # Initialize grouping statistics. This will be automatically loaded from a checkpoint next time.
        self.grouping_stats = torch.nn.Parameter(torch.cat(scales_3d_list))
        self.model.grouping_field.quantile_transformer = self._get_quantile_func(
            torch.cat(scales_3d_list)
        )

        # Turn model back to train mode
        self.model.train()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """
        Same as the base class, but also loads the grouping statistics.
        It's important to normalize the 3D scales as input to the grouping field.
        """
        # Load 3D group scale statistics
        grouping_stats = state_dict["grouping_stats"]
        self.grouping_stats = torch.nn.Parameter(torch.zeros_like(grouping_stats)).to(
            self.device
        )
        # Calculate quantile transformer
        self.model.grouping_field.quantile_transformer = self._get_quantile_func(
            grouping_stats
        )

        return super().load_state_dict(state_dict, strict)

    def _get_quantile_func(self, scales: torch.Tensor, distribution="normal"):
        """
        Use 3D scale statistics to normalize scales -- use quantile transformer.
        """
        scales = scales.flatten()
        scales = scales[(scales > 0) & (scales < self.config.max_grouping_scale)]

        scales = scales.detach().cpu().numpy()

        # Calculate quantile transformer
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

        def quantile_transformer_func(scales):
            # This function acts as a wrapper for QuantileTransformer.
            # QuantileTransformer expects a numpy array, while we have a torch tensor.
            return torch.Tensor(
                quantile_transformer.transform(scales.cpu().numpy())
            ).to(scales.device)

        return quantile_transformer_func
        
    #fron cursor
    def save_training_results(self, max_num_iterations: int):
        """
        Save training results (max GPU memory and total training time) to results.txt
        This should be called when training is complete.
        """
        if not hasattr(self, '_training_start_time'):
            return
        
        try:
            # Calculate total training time
            total_time_seconds = time.time() - self._training_start_time
            hours = int(total_time_seconds // 3600)
            minutes = int((total_time_seconds % 3600) // 60)
            seconds = int(total_time_seconds % 60)
            
            # Get max GPU memory
            max_memory_gb = self._max_gpu_memory_mb / 1024.0 if torch.cuda.is_available() else 0.0
            
            # Find output directory (where config.yml is saved)
            output_dir = None
            outputs_dir = Path("outputs")
            
            if outputs_dir.exists():
                # Find the most recently created config.yml file
                config_files = list(outputs_dir.rglob("config.yml"))
                if config_files:
                    config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    output_dir = config_files[0].parent
            
            # Fallback: use data name if available
            if output_dir is None:
                if hasattr(self, 'datamanager') and hasattr(self.datamanager, 'config'):
                    try:
                        data_name = self.datamanager.config.dataparser.data.name
                        output_dir = Path("outputs") / data_name / "garfield"
                        # Try to find the most recent run directory
                        if output_dir.exists():
                            run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
                            if run_dirs:
                                run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                                output_dir = run_dirs[0]
                    except:
                        output_dir = Path("outputs")
                else:
                    output_dir = Path("outputs")
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write results to file
            results_path = output_dir / "results.txt"
            with open(results_path, "w") as f:
                f.write("Training Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
                f.write(f"Total Training Time (seconds): {total_time_seconds:.2f}\n")
                if torch.cuda.is_available():
                    f.write(f"Max GPU Memory: {max_memory_gb:.2f} GB ({self._max_gpu_memory_mb:.2f} MB)\n")
                else:
                    f.write("Max GPU Memory: N/A (CUDA not available)\n")
                f.write(f"Total Iterations: {max_num_iterations}\n")
                f.write(f"Average Time per Iteration: {total_time_seconds / max_num_iterations:.4f} seconds\n")
            
            print(f"[GARField] Training results saved to {results_path}")
        except Exception as e:
            print(f"[GARField] Warning: Could not save training results: {e}")
    
    def _save_results_on_exit(self):
        """Save results when program exits (called by atexit)"""
        if hasattr(self, '_training_start_time') and not self._results_saved:
            # Use current step as max_num_iterations if we don't know it
            max_iter = self._max_num_iterations if self._max_num_iterations is not None else 0
            self.save_training_results(max_iter)
