import torch
import os
import torch.nn as nn
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch.nn.functional as F
from emo_datasets import *
from model import *
from utils import *


# 1. 配置参数
class Config:
    fusemodel = "attn"
    train_dir = "train"
    val_dir = "val"
    model_name = "model"
    target_layers = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # 选择提取特征的Transformer层
    seq_pool = "end"
    select_seq = 10
    feature_aggregation = "mean"  # 聚合策略: mean/max/concat
    torch_dtype="int8"
    num_classes = 2  # 分类类别数
    batch_size = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "save"
    resume = None
    optim_config = {
        "optimizer": "adamw",
        "epochs": 20, 
        "amsgrad": "False",
        "base_lr": 0.0001,
        "lr_min": 0.000001,
        "betas": [0.9, 0.999],
        "weight_decay": 0.05,
        "scheduler": "cosine",
        'steps_per_epoch': 741
    }
    

def save_classifier_checkpoint(epoch, classifier, optimizer, best_acc, save_dir):
    """
    只保存分类模块的参数和相关训练状态
    """
    checkpoint = {
        'epoch': epoch,
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': best_acc
    }
    
    dirs, _ = os.path.split(save_dir)
    os.makedirs(dirs, exist_ok=True)
    torch.save(checkpoint, save_dir)
    print(f"Saved classifier checkpoint at epoch {epoch}")

# 加载分类模块checkpoint的函数
def load_classifier_checkpoint(model, optimizer, checkpoint_path):
    """
    从checkpoint加载分类模块参数和优化器状态
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # 加载checkpoint[2,6](@ref)
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # 加载分类模块参数
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 恢复训练状态
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"Resuming training from epoch {start_epoch} with best acc {best_acc:.4f}")
    return model, start_epoch, best_acc


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        targets = F.one_hot(targets, num_classes=2).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.sum()

def main(config):
    # model.to(config.device)
    # 初始化处理器和模型
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).cuda())
    # criterion = FocalLoss()
    total_step = 1
    best_acc = 0
    start_epoch = 0
    processor = AutoProcessor.from_pretrained(config.model_name)
    model = QwenAudioClassifier(config)
    optimizer, scheduler = create_optimizer(model.parameters(), config.optim_config)

    if config.resume:
        model, start_epoch, best_acc = load_classifier_checkpoint(model, optimizer, config.resume)
        total_step = 555*start_epoch
        for i in range(start_epoch):
            scheduler.step()
    

    
    train_files, train_labels, valid_files, valid_labels = prepare_binary_data(config.train_dir, config.val_dir)
    train_dataset = BinaryEmoAudioDataset(train_files, train_labels, processor)
    valid_dataset = BinaryEmoAudioDataset(valid_files, valid_labels, processor)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False)

    # 开始训练
    writer = SummaryWriter(f'{config.save_dir}/tensorboard') 


    for epoch in range(start_epoch, config.optim_config['epochs']):
        model.train()
        loss_log = 0
        inner_step = 0
        for inputs, labels in tqdm.tqdm(train_loader):
            # if total_step % 100 > 5:
            #     break
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            labels = labels.to(config.device)
            outputs = model(inputs)

            # import pdb
            # pdb.set_trace()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log += loss.item()
            inner_step += 1
            tqdm.tqdm.write(f"Step {total_step}/{len(train_dataset)//config.batch_size} | Train Loss: {loss_log/inner_step:.4f}")
            if total_step%50==0:
                writer.add_scalar('Loss/train', loss_log/(1+(total_step%(len(train_loader)))), total_step)
            total_step+=1
            scheduler.step()
        # 验证步骤
        valid_loss, acc = evaluate(model, valid_loader, criterion)
        print(f"Epoch {epoch+1}/{config.optim_config['epochs']} | Valid Loss: {valid_loss:.4f} | Acc: {acc:.4f}")
        writer.add_scalar('Accuracy/val', acc, epoch)
        writer.add_scalar('Accuracy/valid_loss', valid_loss, epoch)
        if acc > best_acc:
            best_acc = acc
            save_classifier_checkpoint(epoch, model.classifier, optimizer, best_acc, config.save_dir+f"/best_epoch_{epoch}.pth")
        save_classifier_checkpoint(epoch, model.classifier, optimizer, acc, config.save_dir+f"/latest.pth")

# 7. 评估函数
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader):
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / len(loader), correct / total



if __name__ == "__main__":
    config = Config()
    main(config)