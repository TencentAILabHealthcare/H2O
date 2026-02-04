class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



def save_ddp_checkpoint(epoch, model, optimizer, lr_scheduler, train_loss, 
                       save_to='checkpoints', is_best=False, additional_info=None):
    """
    保存DDP训练检查点（包含lr_scheduler状态）
    """
    import os
    import torch
    from datetime import datetime
    
    os.makedirs(save_to, exist_ok=True)
    
    # 准备检查点数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ddp_training': True
    }
    
    # 保存lr_scheduler状态
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    # 保存最后一次学习率
    checkpoint['current_lr'] = optimizer.param_groups[0]['lr']
    
    # 对于ReduceLROnPlateau，还需要保存最佳值
    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        checkpoint['lr_scheduler_best'] = lr_scheduler.best
        checkpoint['lr_scheduler_num_bad_epochs'] = lr_scheduler.num_bad_epochs
        checkpoint['lr_scheduler_cooldown_counter'] = lr_scheduler.cooldown_counter
    
    # 添加额外信息
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # 保存文件
    filename = f'checkpoint_epoch_{epoch}.pth'
    filepath = os.path.join(save_to, filename)
    torch.save(checkpoint, filepath)
    
    # 保存最新检查点
    latest_path = os.path.join(save_to, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(save_to, 'best_model.pth')
        torch.save(checkpoint, best_path)
    
    print(f"✅ 检查点已保存: {filepath}")
    return filepath

def load_ddp_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, 
                                      device='cuda', strict=True):
    """
    加载检查点并恢复lr_scheduler状态
    """
    import torch
    import os
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return 0, None, {}
    
    try:
        print(f"📂 加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
        
        # 加载模型权重
        state_dict = checkpoint['model_state_dict']
        
        # 处理DDP前缀
        is_ddp_model = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
            if not is_ddp_model:
                state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
        else:
            if is_ddp_model:
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        
        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ 优化器状态已恢复")
            
            # 确保优化器状态在正确设备上
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        
        # 恢复lr_scheduler状态
        if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print("✅ 学习率调度器状态已恢复")
            
            # 对于ReduceLROnPlateau，还需要恢复额外属性
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if 'lr_scheduler_best' in checkpoint:
                    lr_scheduler.best = checkpoint['lr_scheduler_best']
                if 'lr_scheduler_num_bad_epochs' in checkpoint:
                    lr_scheduler.num_bad_epochs = checkpoint['lr_scheduler_num_bad_epochs']
                if 'lr_scheduler_cooldown_counter' in checkpoint:
                    lr_scheduler.cooldown_counter = checkpoint['lr_scheduler_cooldown_counter']
                
                print(f"   ReduceLROnPlateau状态: best={lr_scheduler.best}, "
                      f"num_bad_epochs={lr_scheduler.num_bad_epochs}")
        
        # 检查学习率是否一致
        if 'current_lr' in checkpoint:
            current_lr = optimizer.param_groups[0]['lr']
            saved_lr = checkpoint['current_lr']
            if abs(current_lr - saved_lr) > 1e-6:
                print(f"⚠️  学习率不一致: 保存的={saved_lr}, 当前的={current_lr}")
                # 可以选择恢复保存的学习率
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = saved_lr
        
        # 收集其他信息
        info = {k: v for k, v in checkpoint.items() 
               if k not in ['model_state_dict', 'optimizer_state_dict', 'lr_scheduler_state_dict']}
        
        print(f"✅ 成功加载检查点 (epoch: {epoch}, loss: {loss})")
        return epoch, loss, info
        
    except Exception as e:
        print(f"❌ 加载检查点时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0, None, {}