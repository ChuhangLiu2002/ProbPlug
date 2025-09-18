import torch
import os
import torch.nn as nn
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import pandas as pd
from emo_datasets import *
from model import *
from utils import *
from EmoEval import *

seed = 42
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子

# 1. 配置参数
class Config:
    train_dir = "train"
    val_dir = "val"
    test_dir = "test"
    model_name = "model"
    target_layers = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # 选择提取特征的Transformer层
    seq_pool = "end"
    select_seq = 10
    feature_aggregation = "mean"  # 聚合策略: mean/max/concat
    torch_dtype="int8"
    num_classes = 2  # 分类类别数
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "save"
    resume = "resume"
    optim_config = {
        "optimizer": "adamw",
        "epochs": 20, 
        "amsgrad": "False",
        "base_lr": 0.0001,
        "lr_min": 0.00001,
        "betas": [0.9, 0.999],
        "weight_decay": 0.05,
        "scheduler": "cosine",
        'steps_per_epoch': 556
    }
    
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
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu',),
        weights_only=False
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

def main(config):
    start_epoch = 0
    processor = AutoProcessor.from_pretrained(config.model_name)
    model = QwenAudioClassifier(config)
    optimizer, scheduler = create_optimizer(model.parameters(), config.optim_config)

    if config.resume:
        model, _, _ = load_classifier_checkpoint(model, optimizer, config.resume)
        for i in range(start_epoch):
            scheduler.step()
    
    test_files, test_labels  = handle_binary_json2index(config.test_dir)
    test_dataset = BinaryAudioEvalDataset(test_files, test_labels, processor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    evaluate(model, test_loader)

# 7. 评估函数
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    label_list = []
    predictions = []
    probs = []
    with torch.no_grad():
        for inputs, labels, ff in tqdm.tqdm(loader):
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            res = nn.Softmax(dim=1)(outputs)
            prob = res[0][1].item()
            # import pdb;pdb.set_trace()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.append({"key":ff, "pred":predicted.item()})
            probs.append({"key":ff, "prob":prob})
            label_list.append({"key":ff, "label":labels.item()})


    emo_eval = EmoEval(predictions, label_list)
    fold_scores = emo_eval.compute_metrics()

    
    outputs_pred = [(i['key'], i['prob'], j['label']) for i, j in zip(probs, label_list)]
    out = {'key':[i[0] for i in outputs_pred], 'prob':[i[1] for i in outputs_pred], 'label':[i[2] for i in outputs_pred]}
    df = pd.DataFrame(out)
    df.to_csv(f"{config.save_dir}/best_pred_scores_train.csv", index=False)
    
    # 修改：添加"fine"前缀
    emo_eval.write_scores(f"{config.save_dir}/best_binary_scores_train.txt", fold_scores)
    print(f"Fold {1} 二分类评估完成")
    return total_loss / len(loader), correct / total



if __name__ == "__main__":
    config = Config()
    main(config)