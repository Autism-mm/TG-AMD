import argparse
import torch
import psutil
import os
import torch
import collections
from tqdm import tqdm
from utils.dataload import *
from models import *
from utils.traintest import model_train, model_val,model_test,model_train3
import numpy as np
from Model import *

import time
import random
def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    # parser.add_argument('--data_path', type=str, default="/data/lqc/msb/EOC_scene/",
    #                     help='where data is stored')
    parser.add_argument('--data_path', type=str, default="../SARATR_DDPM/SOC_40classes_test/",
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=str, default='0',
                    help='Comma-separated GPU ids, e.g., 0,1,2')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=1,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='directory to save models')
    args = parser.parse_args()
    return args
def print_memory_usage():
    print(f"CPU Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
from strengthen_image import create_argparser
from diffusion_utils.script_util import *
if __name__ == '__main__':
    arg = parameter_setting()
    
    # 创建保存模型的目录
    os.makedirs(arg.save_dir, exist_ok=True)
    print("PyTorch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    device_ids = [int(x) for x in str(arg.GPU_ids).split(',')]
    torch.cuda.set_device(device_ids[0]) 
    history = collections.defaultdict(list)
    train_all,num_traclasses = load_data(arg.data_path + 'train', data_transform)
    print(f"训练集大小: {len(train_all)}")
    if len(train_all) > 0:
        first_image, _ = train_all[0]
        print(f"图片形状: {first_image.shape}")
    test_set,num_testclasses = load_data(arg.data_path + 'test', data_transform)
    
    for k_F in tqdm(range(arg.fold)):
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)
        model = convnext_1(num_traclasses)
        model = model.cuda() 
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        args = create_argparser().parse_args()
        model2, diffusion = sr_create_model_and_diffusion(
            **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
        )
        model2 = model2.cuda() 
        model2.convert_to_fp16()  # 先转换为 FP16
        model2 = torch.nn.DataParallel(model2, device_ids=device_ids)  # 再包装

        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-4, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
        best_val_accuracy = 0
        
        for epoch in range(1, arg.epochs + 1):
            print(f"##### Fold {k_F + 1}, Epoch {epoch}/{arg.epochs} #####")
            # print_memory_usage()
            final_acc = model_train3(model=model, data_loader=train_loader, opt=opt, sch=scheduler,model2=model2,diff=diffusion,args=args)
            
            # 验证和保存频率调整：前50个epoch每50个epoch验证，50-100每10个epoch验证，>100每5个epoch验证
            if (epoch <= 50 and (epoch % 50 == 0)) or (epoch > 50 and epoch <= 100 and (epoch % 10 == 0)) or (epoch > 100 and (epoch % 5 == 0)) or (epoch == arg.epochs):
                val_accuracy = model_val(model, test_loader,model2=model2,diff=diffusion,args=args)
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
                # 如果当前验证准确率是最好的，保存模型
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_accuracy': val_accuracy,
                        'fold': k_F + 1
                    }
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = os.path.join(arg.save_dir, f'best_model_fold{k_F+1}_{timestamp}.pth')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved best model with validation accuracy: {val_accuracy:.2f}%")
        
        print(f"\nFold {k_F+1} completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # 在所有fold完成后，找出最佳模型
    best_fold_acc = 0
    best_fold = 0
    for k_F in range(arg.fold):
        checkpoint_path = os.path.join(arg.save_dir, f'best_model_fold{k_F+1}.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if checkpoint['val_accuracy'] > best_fold_acc:
                best_fold_acc = checkpoint['val_accuracy']
                best_fold = k_F + 1
    
    print(f"\nTraining completed!")
    print(f"Best overall model is from fold {best_fold} with validation accuracy: {best_fold_acc:.2f}%")
    print(f"Model saved at: {os.path.join(arg.save_dir, f'best_model_fold{best_fold}.pth')}")

#先不保存，之后可以保存，class自己识别，这还得自己调，太麻烦了。