import argparse
from diffusion_utils.script_util import *
from diffusion_utils.unet import *
from diffusion_utils.image_datasets import load_data
import torch
from diffusion_utils.resample import *
from diffusion_utils.train_utils import *

# 将数据集类移到模块级别
class TempDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device):
        self.x = x
        self.y = y
        self.device = device
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x_item = self.x[idx]
        # 返回格式要匹配 ImageDataset 的输出格式
        out_dict = {
            "SR": x_item,  # 输入图像作为 SR 条件
            "HR": x_item   # 同样的图像作为高分辨率目标
        }
        return x_item, out_dict

def strengthen_diffusion(model2,diffusion,x_rowremove,y_rowremove,args):
    # print(np.shape(x_rowremove),'x_rowremove')
    # print(np.shape(y_rowremove),'y_rowremove')
    device = torch.device(f"{args.device}:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 确保模型在正确的设备上
    model2 = model2.to(device)
    
    # 将数据移到 GPU
    x_rowremove = x_rowremove.to(device)
    y_rowremove = y_rowremove.to(device)
    
    # 保存原始输入，用于后续采样
    original_x = x_rowremove.clone()
    
    dataset = TempDataset(x_rowremove, y_rowremove, device)
    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # 设置为0以避免多进程引起的设备问题
    )
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    # 设置训练步数
    if not hasattr(args, 'max_train_steps'):
        args.max_train_steps = 100  # 默认训练100步
    
    # 设置 lr_anneal_steps 来控制训练步数
    args.lr_anneal_steps = args.max_train_steps
    
    # 训练模型
    TrainLoop(
        model=model2,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()
    
    # 使用训练好的模型进行采样
    model2.eval()
    with torch.no_grad():
        # 使用扩散模型的采样方法生成去噪后的图像
        sample = diffusion.p_sample_loop(
            model2,
            original_x.shape,
            clip_denoised=True,
            model_kwargs={"SR": original_x, "y": y_rowremove}
        )
    
    return sample, y_rowremove  # 返回去噪后的图像和原始标签

def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        batch_size=2,
        num_workers=4,
        device="cuda",
        gpu_id=0,
        max_train_steps=10,  # 添加最大训练步数参数
        # 添加模型参数
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        class_cond=True,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=True,
        # 添加扩散模型参数
        diffusion_steps=100,
        noise_schedule="linear",
        timestep_respacing="ddim100",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        microbatch=1,
        ema_rate="0.9999",
        fp16_scale_growth=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

