import torch
import os
import torch
import torch.nn as nn
# 假设你的模型是 model，优化器是 optimizer，当前的 epoch 是 epoch，损失是 loss
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    # 创建 checkpoint 文件夹（如果不存在）
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 定义保存 checkpoint 的文件名
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    # 保存模型参数、优化器状态和其他训练信息
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded (epoch {epoch}, loss {loss})")
        return model, optimizer, epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return model, optimizer, 0, 0.0