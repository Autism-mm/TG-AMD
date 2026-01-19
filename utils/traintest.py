import torch
import torch
import psutil
import os
from torch import nn
import numpy as np
import torch.nn.functional as F
from strengthen_image import strengthen_diffusion
import warnings
from torch.cuda.amp import GradScaler, autocast

# 过滤掉命名张量的警告
warnings.filterwarnings('ignore', message='Named tensors.*')

def print_memory_usage():
    print(f"CPU Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

def custom_loss(preds, targets, temperature=1.0):
    # 计算每个样本的最大概率
    probs = F.softmax(preds, dim=1)
    max_probs = probs.max(dim=1)[0]  # shape: [batch_size]
    # 计算权重，权重范围在[1.0, 1.0 + temperature]
    weights = 1.0 + temperature * (1.0 - max_probs)
    # 计算基础交叉熵损失
    base_loss = F.cross_entropy(preds, targets, reduction='none')  # shape: [batch_size]
    # 应用权重
    weighted_loss = weights * base_loss
    return weighted_loss.mean()

def model_train3(model, data_loader, opt, sch, model2, diff, args):#半精度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # 初始化 GradScaler
    scaler = GradScaler()
    
    total_correct = 0
    total_samples = 0
    
    for i, data in enumerate(data_loader):
        # 清理之前的缓存
        torch.cuda.empty_cache()
        
        x, y = data
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)
        
        # 1. 对每个批次随机选择一部分样本进行增强
        enhance_mask = (torch.rand(batch_size) < 0.3).to(device)
        if enhance_mask.any():
            x_to_enhance = x[enhance_mask]
            y_to_enhance = y[enhance_mask]
            
            # 在增强之前清理缓存
            torch.cuda.empty_cache()
            
            # 增强过程
            x_enhanced, _ = strengthen_diffusion(model2, diff, x_to_enhance, y_to_enhance, args)
            x_enhanced = x_enhanced.to(device)
            
            # 清理不需要的中间变量
            del x_to_enhance, y_to_enhance
            torch.cuda.empty_cache()
            
            # 将原始样本和增强样本混合
            x_mixed = x.clone()
            x_mixed[enhance_mask] = x_enhanced
            # 使用自动混合精度
            opt.zero_grad()
            with autocast():
                # print_memory_usage()
                # 计算混合损失
                output_orig = model(x)
                loss_orig = custom_loss(output_orig, y, temperature=0.5)
                # print_memory_usage()
                # 计算增强样本的损失
                output_mixed = model(x_mixed)
                # print_memory_usage()
                # print('5')
                loss_mixed = custom_loss(output_mixed, y, temperature=0.5)
                
                # 计算总损失
                loss = (loss_orig + loss_mixed) / 2
                
                # 添加一致性损失
                consistency_loss = F.mse_loss(output_orig[enhance_mask], output_mixed[enhance_mask])
                loss = loss + 0.1 * consistency_loss
            
            # 使用 scaler 进行反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            
            # 计算准确率（使用混合后的预测结果）
            with torch.no_grad():
                pred_mixed = output_mixed.max(1, keepdim=True)[1]
                total_correct += pred_mixed.eq(y.view_as(pred_mixed)).sum().item()
            
            # 清理不需要的张量
            del output_orig, output_mixed, x_enhanced, x_mixed, consistency_loss
            torch.cuda.empty_cache()
            
        else:
            # 使用自动混合精度
            opt.zero_grad()
            with autocast():
                output = model(x)
                loss = custom_loss(output, y, temperature=0.5)
            
            # 使用 scaler 进行反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            
            # 计算准确率
            with torch.no_grad():
                pred = output.max(1, keepdim=True)[1]
                total_correct += pred.eq(y.view_as(pred)).sum().item()
            
            # 清理不需要的张量
            del output
            torch.cuda.empty_cache()
        
        # 清理loss
        del loss
        torch.cuda.empty_cache()
        
        # 累计样本数
        total_samples += batch_size
        
        # 每10个批次打印一次当前准确率和内存使用情况
        if (i + 1) % 10 == 0:
            current_acc = 100. * total_correct / total_samples
            print(f"Batch [{i+1}/{len(data_loader)}], Current Training Accuracy: {current_acc:.2f}%")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    # 计算整体训练准确率
    final_acc = 100. * total_correct / total_samples
    print(f"Final Training Accuracy: {final_acc:.2f}%")
    
    # 最后清理一次
    torch.cuda.empty_cache()
    
    return final_acc
def model_train(model, data_loader, opt, sch, model2, diff, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_acc = 0
    cr1 = nn.CrossEntropyLoss()
    correct = 0
    # print("middle loading VGG19")
    # print_memory_usage()
    
    # 用于收集增强样本
    
    
    # 第一阶段：使用原始数据训练，同时收集需要增强的样本
    for i, data in enumerate(data_loader):
        x, y = data
        x = x.to(device)  # 确保数据在正确的设备上
        y = y.to(device)
        
        # 原始数据的前向传播
        output = model(x)
        
        # 使用自定义损失函数，temperature=0.5 使权重范围在[1.0, 1.5]
        loss = custom_loss(output, y, temperature=0.5)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        
        # 统计原始数据的准确率
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        
        if (i + 1) % 100 == 0:
            current_acc = 100. * correct / ((i + 1) * data_loader.batch_size)
            print(f"Batch [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}, Current Accuracy: {current_acc:.2f}%")
    
    final_acc = 100. * correct / len(data_loader.dataset)
    print(f"Final Training Accuracy: {final_acc:.2f}%")
    return final_acc

def model_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print("Test Accuracy is:{:.2f} %: ".format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def model_val(model, test_loader, model2, diff, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 评估模式
    correct = 0
    correct_enhanced = 0
    total_samples = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total_samples += len(data)
        print(total_samples)
        # 原始验证
        with torch.no_grad():
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 找出低置信度样本
            output_softmax = F.softmax(output, dim=1)
            row_max = output_softmax.max(dim=1)[0]
            mask = row_max < 0.9
            rows_to_enhance = mask.nonzero(as_tuple=True)[0]
        
        # 对低置信度样本进行增强 (在no_grad上下文之外)
        if len(rows_to_enhance) > 0:
            x_to_enhance = data[rows_to_enhance].clone()
            y_to_enhance = target[rows_to_enhance].clone()
            
            # 临时将diffusion模型切换到训练模式
            model2.train()
            # print_memory_usage()
            try:
                # 获取增强后的样本
                x_enhanced, _ = strengthen_diffusion(model2, diff, x_to_enhance, y_to_enhance, args)
            finally:
                # 恢复到评估模式
                model2.eval()
            
            # 使用增强样本进行预测
            with torch.no_grad():
                output_enhanced = model(x_enhanced.to(device))
                pred_enhanced = output_enhanced.max(1, keepdim=True)[1]
                
                # 更新正确数（只更新原来预测错误的部分）
                original_pred = pred[rows_to_enhance]
                original_correct = original_pred.eq(y_to_enhance.view_as(original_pred))
                enhanced_correct = pred_enhanced.eq(y_to_enhance.view_as(pred_enhanced))
                
                # 如果原来预测错误，但增强后预测正确，则更新correct
                improvements = (~original_correct & enhanced_correct).sum().item()
                correct_enhanced = correct + improvements
                
                # if improvements > 0:
                #     print(f"增强后正确预测增加了 {improvements} 个样本")
    
    # 计算原始准确率和增强后的准确率
  
    original_acc = 100. * correct / total_samples
    enhanced_acc = 100. * correct_enhanced / total_samples
    
    print(f"Original Validation Accuracy: {original_acc:.2f}%")
    print(f"Enhanced Validation Accuracy: {enhanced_acc:.2f}%")
    
    return enhanced_acc  # 返回增强后的准确率