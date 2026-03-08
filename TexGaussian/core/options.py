import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional, List


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    input_depth: int = 8
    full_depth: int = 4
    in_channels: int = 3
    out_channels: int = 7
    input_feature: str = 'ND'
    # Unet definition
    model_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 8)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True)
    mid_attention: bool = True
    up_attention: Tuple[bool, ...] = (True, True, False, False, False)
    num_heads: int = 16
    context_dim: int = 768
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 128
    # gaussian render size
    output_size: int = 512
    use_material: str = 'True'
    # whether to add supervision for material branch during training
    supervise_material: bool = False
    gaussian_loss: str = 'False'
    use_text: str = 'True'
    use_local_pretrained_ckpt: str = 'False'
    text_description: str = 'Cap3D_automated_Objaverse_full.csv'
    ema_rate: float = 0.999
    radius: float = 1/2
    use_checkpoint: str = 'True'

    ## fit gaussians
    gaussian_list: str = 'pbr_train_list_gaussian.txt'
    mean_path: str = 'statistics/gaussian_mean.pth'
    std_path: str = 'statistics/gaussian_std.pth'

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    fovy: float = 30
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 8
    total_num_views: int = 64
    reference_num_views: int = 15
    # num workers
    num_workers: int = 1
    reference_image_mode: str = 'albedo'
    trainlist: str = 'pbr_train_list.txt'
    testlist: str = 'pbr_train_list.txt'
    image_dir: str = 'path_to_image_dir'
    pointcloud_dir: str = 'path_to_pointcloud_dir'
    gaussian_dir: str = 'path_to_fitted_gaussian'

    # coordinate transform (for mesh/points/normals)
    # swap Y and Z axis
    coord_swap_yz: bool = True
    # flip axes
    coord_flip_x: bool = False
    coord_flip_y: bool = False
    coord_flip_z: bool = True

    # mesh normalization control
    # if True, scale mesh by max radius to unit sphere; if False, only recenter
    normalize_to_unit: bool = False

    # camera control for grid training/inference visualization
    # yaw offset in degrees (e.g., 90 for left-rotate 90°)
    grid_yaw_offset_deg: float = 0.0
    # camera radius override for grid cameras; if <=0, fallback to grid_cam_radius
    grid_cam_radius_override: float = 1.7
    # flip LR for camera yaws (mirror horizontally at camera level)
    grid_flip_lr: bool = False

    ### texture baking
    text_prompt: str = ''
    texture_cam_radius: float = 4.5
    texture_name: str = 'test'
    save_image: str = 'False'
    num_gpus: int = 8
    workers_per_gpu: int = 1
    gpu_ids: str = '[0,1,2,3,4,5,6,7]'
    mesh_path: str = '/workspace/wudang_vuc_3dc_afs/wuchenming/bpfs/pbr_obj_cvpr'
    output_dir: str = 'texture_mesh'
    ckpt_path: str = ''
    # optional: flip mesh render horizontally in texture baking
    texture_flip_horizontal: bool = True

    ### grid training (custom data)
    # directory containing grid_{id}.png
    grid_image_dir: str = ''
    # root containing per-id folder with mesh.obj, e.g., mesh_root/{id}/mesh.obj
    mesh_root: str = ''
    # text file with one prompt per line; line index == id
    text_file: str = ''
    # number of points to sample from mesh surface per item
    num_points: int = 100000
    # camera settings for 4-view supervision
    grid_cam_radius: float = 1.7
    grid_fovy: float = 39.6

    ### training
    # workspace
    workspace: str = 'workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 8
    # ckpt interval
    ckpt_interval: int = 1
    # image interval
    image_interval: int = 100
    # gradient accumulation
    gradient_accumulation_steps: int = 2
    # training epochs
    num_epochs: int = 1000
    # lpips loss weight
    lambda_lpips: float = 1.0
    # ssim loss weight (foreground-weighted)
    lambda_ssim: float = 0.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 4e-4
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    # debugging: overfit single sample
    overfit_one: bool = False
    overfit_id: int = 1668

    # color/CLIP auxiliary losses
    lambda_color: float = 0.0  # masked color MSE weight
    lambda_clip: float = 1.0   # adapt loss: CLIP random-crop similarity weight
    fine_match_weight: float = 1.0  # match loss: DINO fine-grained one-to-one matching weight
    fine_match_target_q: int = 672  # feature extraction resize for fine match
    fine_match_white_bg: bool = True  # mask pure white background in fine match
    # downweight white background contribution in color/ssim losses (0: off, 1: full downweight by whiteness)
    bg_downweight: float = 0.0
    clip_num_crops: int = 8
    clip_min_scale: float = 0.4
    clip_max_scale: float = 1.0
    clip_input_size: int = 224

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = True
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    # training mode: use only LPIPS per-view loss with white background (for grid dataset)
    lpips_only: bool = False

    # debugging
    debug_log: bool = False
    debug_interval: int = 100


# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['objaverse'] = 'the default settings of Objaverse'
config_defaults['objaverse'] = Options()

config_doc['shapenet'] = 'the default settings of shapenet'
config_defaults['shapenet'] = Options(
    use_material = 'False',
    use_text = 'False',
    model_channels=32,
    down_attention = (False, False, False, False, False),
    mid_attention = False,
    up_attention = (False, False, False, False, False),
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
