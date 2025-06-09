"""
Nerfstudio InstructGS2GS Pipeline
"""

import matplotlib.pyplot as plt
import pdb
import typing
from dataclasses import dataclass, field
from itertools import cycle
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

#eventually add the igs2gs datamanager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig,FullImageDatamanager
from igs2gs.igs2gs import InstructGS2GSModel,InstructGS2GSModelConfig
from igs2gs.igs2gs_datamanager import InstructGS2GSDataManagerConfig

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from igs2gs.ip2p import InstructPix2Pix

# imports for secret view editing
import yaml
from types import SimpleNamespace
from torchvision.utils import save_image

from igs2gs.ip2p_ptd import IP2P_PTD

from igs2gs.ip2p_depth import InstructPix2Pix_depth

from nerfstudio.engine.callbacks import TrainingCallbackAttributes


@dataclass
class InstructGS2GSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""
    
    _target: Type = field(default_factory=lambda: InstructGS2GSPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = InstructGS2GSDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = InstructGS2GSModelConfig()
    """specifies the model config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 12.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    gs_steps: int = 2500
    """how many GS steps between dataset updates"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.7
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = False
    """Whether to use full precision for InstructPix2Pix"""

class InstructGS2GSPipeline(VanillaPipeline):
    """InstructGS2GS Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """
    
    def __init__(
        self,
        config: InstructGS2GSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        
        # # select device for InstructPix2Pix
        # self.ip2p_device = (
        #     torch.device(device)
        #     if self.config.ip2p_device is None
        #     else torch.device(self.config.ip2p_device)
        # )

        # self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        # # load base text embedding using classifier free guidance
        # self.text_embedding = self.ip2p.pipe._encode_prompt(
        #     self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        # )

        # which image index we are editing
        self.curr_edit_idx = 0
        # whether we are doing regular GS updates or editing images
        self.makeSquentialEdits = False

        config_path = 'config/config.yaml'
        with open(config_path, 'r') as file:
            cfg_dict = yaml.safe_load(file)
        self.config_secret = SimpleNamespace(**cfg_dict)

        self.config_secret.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        self.ip2p_ptd = IP2P_PTD(
            self.dtype, 
            self.config_secret.device, 
            prompt=self.config_secret.prompt_2, 
            t_dec=self.config_secret.t_dec, 
            image_guidance_scale=self.config_secret.image_guidance_scale_ip2p_ptd, 
            async_ahead_steps=self.config_secret.async_ahead_steps
        )

        self.text_embeddings_ip2p = self.ip2p_ptd.text_embeddings_ip2p

        # ip2p = InstructPix2Pix(device)
        self.ip2p_depth = InstructPix2Pix_depth(
            self.dtype, 
            self.config_secret.device, 
            self.config_secret.render_size, 
            self.config_secret.conditioning_scale
        )

        # secret data preparation
        secret_view_idx = self.config_secret.secret_view_idx
        self.camera_secret, self.data_secret = self.datamanager.next_train_idx(secret_view_idx)
        self.original_image_secret = self.datamanager.original_cached_train[secret_view_idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        self.depth_image_secret = self.datamanager.original_cached_train[secret_view_idx]["depth"] # [bs, h, w]
    
    # add callback function to fetch the components from other parts of the training process.
    def get_training_callbacks(self, attrs: TrainingCallbackAttributes):
        # stash a reference to the Trainer
        self.trainer = attrs.trainer
        # now return whatever callbacks the base class wants
        return super().get_training_callbacks(attrs)

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        ckpt_dir = self.trainer.checkpoint_dir
        image_dir = ckpt_dir / "images"
        if not image_dir.exists():
            image_dir.mkdir(parents=True, exist_ok=True)

        # print(step) # start from 30000
    
        # if ((step-1) % self.config.gs_steps) == 0:
        if (step % self.config.gs_steps) == 0: # update also for the first step
            self.makeSquentialEdits = True

        if (not self.makeSquentialEdits):
            camera, data = self.datamanager.next_train(step)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)
            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            if step % 500 == 0:
                rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                save_image((rendered_image).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

            # also update the secret view w/o editing
            model_outputs_secret = self.model(self.camera_secret)
            metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
            loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

            for k, v in metrics_dict_secret.items():
                metrics_dict[f"secret_{k}"] = v
            for k, v in loss_dict_secret.items():
                loss_dict[f"secret_{k}"] = v

            if step % 500 == 0:
                # save the secret view image
                rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                image_save_secret = torch.cat([rendered_image_secret, self.original_image_secret.to(self.config_secret.device)])
                save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

        else:
            # get index
            idx = self.curr_edit_idx
            camera, data = self.datamanager.next_train_idx(idx)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
            rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

            depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

            # edited_image = self.ip2p.edit_image(
            #             self.text_embedding.to(self.ip2p_device),
            #             rendered_image.to(self.ip2p_device),
            #             original_image.to(self.ip2p_device),
            #             guidance_scale=self.config.guidance_scale,
            #             image_guidance_scale=self.config.image_guidance_scale,
            #             diffusion_steps=self.config.diffusion_steps,
            #             lower_bound=self.config.lower_bound,
            #             upper_bound=self.config.upper_bound,
            #         )
            
            # edit image using IP2P depth when idx != secret_view_idx
            if (idx != self.config_secret.secret_view_idx):
                edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
                    self.text_embeddings_ip2p.to(self.config_secret.device),
                    rendered_image.to(self.dtype),
                    original_image.to(self.config_secret.device).to(self.dtype),
                    False, # is depth tensor
                    depth_image,
                    guidance_scale=self.config_secret.guidance_scale,
                    image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
                    diffusion_steps=self.config_secret.t_dec,
                    lower_bound=self.config_secret.lower_bound,
                    upper_bound=self.config_secret.upper_bound,
                )

                # resize to original image size (often not necessary)
                if (edited_image.size() != rendered_image.size()):
                    edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

                # write edited image to dataloader
                edited_image = edited_image.to(original_image.dtype)
                self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
                data["image"] = edited_image.squeeze().permute(1,2,0)

                # save edited image
                if step % 50 == 0:
                    image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
                    save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

            # ----------- update the secret view dataset ----------- 
            model_outputs_secret = self.model(self.camera_secret)
            metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
            loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
            rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
            edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
                image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
                image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
                depth=self.depth_image_secret,
                lower_bound=self.config_secret.lower_bound,
                upper_bound=self.config_secret.upper_bound
            )

            # resize to original image size (often not necessary)
            if (edited_image_secret.size() != rendered_image_secret.size()):
                edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

            # write edited image to dataloader
            edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
            self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
            self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

            for k, v in metrics_dict_secret.items():
                metrics_dict[f"secret_{k}"] = v
            for k, v in loss_dict_secret.items():
                loss_dict[f"secret_{k}"] = v

            # save edited image
            if step % 50 == 0:
                image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
                save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')
            # ----------- update the secret view dataset ----------- 

            #increment curr edit idx
            # and update all the images in the dataset
            self.curr_edit_idx += 1
            # self.makeSquentialEdits = False
            if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
                self.curr_edit_idx = 0
                self.makeSquentialEdits = False

        return model_outputs, loss_dict, metrics_dict
    
    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
    