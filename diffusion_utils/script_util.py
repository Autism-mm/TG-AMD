from .unet import *
import argparse
NUM_CLASSES = 4
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
import inspect
def sr_create_model_and_diffusion(
    num_channels=128,        # 从 model_and_diffusion_defaults 中的默认值
    num_res_blocks=2,        # 从 model_and_diffusion_defaults 中的默认值
    num_heads=4,             # 从 model_and_diffusion_defaults 中的默认值
    num_head_channels=-1,    # 从 model_and_diffusion_defaults 中的默认值
    num_heads_upsample=-1,   # 从 model_and_diffusion_defaults 中的默认值
    attention_resolutions="16,8",  # 从 model_and_diffusion_defaults 中的默认值
    dropout=0.0,             # 从 model_and_diffusion_defaults 中的默认值
    class_cond=True,         # 设置为True，因为需要使用类别信息
    use_checkpoint=True,     # 从 model_and_diffusion_defaults 中的默认值
    use_scale_shift_norm=True,  # 从 model_and_diffusion_defaults 中的默认值
    resblock_updown=False,   # 从 model_and_diffusion_defaults 中的默认值
    use_fp16=False,         # 从 model_and_diffusion_defaults 中的默认值
    diffusion_steps=1000,    # 从 diffusion_defaults 中的默认值
    noise_schedule="linear", # 从 diffusion_defaults 中的默认值
    timestep_respacing="ddim100",  # 从 diffusion_defaults 中的默认值
    use_kl=False,           # 从 diffusion_defaults 中的默认值
    predict_xstart=False,    # 从 diffusion_defaults 中的默认值
    rescale_timesteps=True,  # 从 diffusion_defaults 中的默认值
    rescale_learned_sigmas=False,  # 从 diffusion_defaults 中的默认值
):
    model = sr_create_model(
        large_size=224,      # 固定为224，因为输入是224x224
        small_size=224,      # 固定为224，因为输入是224x224
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        learn_sigma=False,   # 固定为False，因为不需要学习sigma
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=False,   # 固定为False
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    
    return model, diffusion

def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 224:  # 添加 224 的情况
        channel_mult = (1, 1, 2, 2, 4, 4) 
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=3,  # 修改：固定为3，因为不需要learn_sigma
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),  # 保留：因为我们需要类别条件
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )

#创建高斯扩散模型,将噪声图片变得清晰
def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 256
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="ddim100",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
