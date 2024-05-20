# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=3,
    image_size=(None, None),
    transform_name="resize_crop",
)
# IMG: 1024 (20%) 512 (30%) 256 (50%) drop (50%)
bucket_config = {  # 1s/it
    # "144p": {6: (0.5, 16)},
    # "144p": {1: (0.5, 6)},
    # "144p": {24: (0.5, 1)},
    # "360p": {4: (0.3, 1),},
    # "512": {1: (0.4, 12)},
    # "1024": {1: (0.3, 3)},
}
mask_ratios = {
    "mask_no": 0.75,
    "mask_quarter_random": 0.025,
    "mask_quarter_head": 0.025,
    "mask_quarter_tail": 0.025,
    "mask_quarter_head_tail": 0.05,
    "mask_image_random": 0.025,
    "mask_image_head": 0.025,
    "mask_image_tail": 0.025,
    "mask_image_head_tail": 0.05,
}

# Define acceleration
num_workers = 1
num_bucket_build_workers = 2
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    # type="STDiT2-S1-2",
    type="STDiT2-XL/2",
    # from_pretrained=None,
    input_sq_size=32,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="pretrained_models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="pretrained_models/DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs/stage1_tmp"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 500
load = None

batch_size = None
lr = 2e-5
grad_clip = 1.0
