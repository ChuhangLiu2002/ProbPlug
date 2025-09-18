import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class FeatureProjector(nn.Module):
    def __init__(self, layer_dims, target_dim=1024):
        self.projectors = nn.ModuleList([
            nn.Linear(dim, target_dim) for dim in layer_dims
        ])
    
    def forward(self, hidden_states):
        # hidden_states: 各层特征列表 [ (B, L, D₁), (B, L, D₂), ... ]
        projected = [proj(h) for h, proj in zip(hidden_states, self.projectors)]
        return torch.stack(projected, dim=1)  # → (B, N, L, target_dim)

class sattn_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = 4096
        target_dim=4096
        num_heads = 8
        self.projector_seq = nn.Sequential(
            nn.Linear(config.select_seq, config.select_seq*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.select_seq*2, 1),
            nn.Dropout(0.5),
        ).cuda()
        self.query = nn.Parameter(torch.randn(1, target_dim)).cuda()  # 可学习查询向量
        self.multihead_attn = nn.MultiheadAttention(target_dim, num_heads, batch_first=True).cuda()
        self.norm = nn.BatchNorm1d(1).cuda()
        self.classifier = nn.Sequential(
                                            nn.Linear(input_dim, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(512, config.num_classes)
                                        ).cuda()
    
    def forward(self, features):
        # features = torch.stack(features, dim=1)[:,:, -10:, :].float()
        # 序列聚合
        features = features.permute(0,1,3,2)
        features = self.projector_seq(features).squeeze(-1)

        # 层间聚合
        query = self.query.expand(features.size(0), 1, -1)  # (B, 1, D)
        attn_output, attn_weights = self.multihead_attn(query, features, features)  # Q, K, V
        # import pdb
        # pdb.set_trace()
        attn_output = self.norm(attn_output)
        # classification
        logit = self.classifier(attn_output.squeeze(1))
        return logit


class valina_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = 4096
        target_dim=4096
        num_heads = 8
        self.projector_seq = nn.Sequential(
            nn.Linear(config.select_seq, config.select_seq*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.select_seq*2, 1)
        ).cuda()
        self.query = nn.Parameter(torch.randn(1, target_dim)).cuda()  # 可学习查询向量
        self.multihead_attn = nn.MultiheadAttention(target_dim, num_heads, batch_first=True).cuda()
        self.classifier = nn.Sequential(
                                            nn.Linear(input_dim, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.3),
                                            nn.Linear(512, 4)
                                        ).cuda()
    
    def forward(self, features):
        # 序列聚合
        features = torch.stack(features, dim=1)[:, :, -10:, :].float()
        features = features.permute(0,1,3,2)
        features = self.projector_seq(features).squeeze(-1)

        # 层间聚合
        query = self.query.expand(features.size(0), 1, -1)  # (B, 1, D)
        attn_output, attn_weights = self.multihead_attn(query, features, features)  # Q, K, V
        
        # classification
        logit = self.classifier(attn_output.squeeze(1))
        return logit
    

class dualQformer_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = 4096
        target_dim=4096
        num_heads = 8
        # self.projector_seq = nn.Sequential(
        #     nn.Linear(config.select_seq, config.select_seq*2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(config.select_seq*2, 1)
        # ).cuda()
        self.query_seq = nn.Parameter(torch.randn(1, config.select_seq)).cuda()  # 可学习查询向量
        self.multihead_attn_seq = nn.MultiheadAttention(config.select_seq, 2, batch_first=True).cuda()

        self.query = nn.Parameter(torch.randn(1, target_dim)).cuda()  # 可学习查询向量
        self.multihead_attn = nn.MultiheadAttention(target_dim, num_heads, batch_first=True).cuda()
        self.classifier = nn.Sequential(
                                            nn.Linear(input_dim, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.3),
                                            nn.Linear(512, 4)
                                        ).cuda()
    
    def forward(self, features):
        # 序列聚合
        import pdb
        pdb.set_trace()
        features = features.permute(0,1,3,2)
        query = self.query.expand(features.size(0), 1, -1)  # (B, 1, D)
        attn_output, attn_weights = self.multihead_attn(query, features, features)  # Q, K, V
        features = self.projector_seq(features).squeeze(-1)

        # 层间聚合
        query = self.query.expand(features.size(0), 1, -1)  # (B, 1, D)
        attn_output, attn_weights = self.multihead_attn(query, features, features)  # Q, K, V
        
        # classification
        logit = self.classifier(attn_output.squeeze(1))
        return logit
    


class QwenAudioClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(config.torch_dtype)
        if self.config.torch_dtype == 'int8':
            self.extractor = Qwen2AudioForConditionalGeneration.from_pretrained(config.model_name,
                                                                                # torch_dtype="bfloat16",
                                                                                load_in_8bit=True,
                                                                                device_map="cuda:0",  # 自动分配设备
                                                                                trust_remote_code=True)  # 允许加载远程代码)
        elif self.config.torch_dtype == 'int4':
            self.extractor = Qwen2AudioForConditionalGeneration.from_pretrained(config.model_name,
                                                                                load_in_4bit=True,
                                                                                device_map="cuda:0",  # 自动分配设备
                                                                                trust_remote_code=True)  # 允许加载远程代码)
        else:
            raise "Check the torch type"
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.extractor.eval()
        
        # 冻结QWen-Audio参数
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.classifier = sattn_Classifier(config)
        # self.classfier = dualQformer_Classifier(config)
    
    
    def forward(self, audio_inputs):
        # obtain features
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            outputs = self.extractor(
                **audio_inputs,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = outputs.hidden_states
        all_ones_mask = torch.sum(outputs['attention_mask'], dim=1).tolist()


        selected_features = [hidden_states[i] for i in self.config.target_layers]
        features = torch.stack(selected_features, dim=1).float()

        features = torch.stack([features[idx, :, i-10:i, :] for idx, i in enumerate(all_ones_mask)])
        return self.classifier(features)



class AudioT_QwenAudioClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.extractor = Qwen2AudioForConditionalGeneration.from_pretrained(config.model_name,
                                                                            # torch_dtype="bfloat16",
                                                                            load_in_8bit=True,
                                                                            device_map="auto",  # 自动分配设备
                                                                            trust_remote_code=True)  # 允许加载远程代码)
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.extractor.eval()
        
        # 冻结QWen-Audio参数
        for param in self.extractor.parameters():
            param.requires_grad = False

        # self.classifier = sattn_Classifier(config)
        self.classifier = dualQformer_Classifier(config)
    
    
    def forward(self, audio_inputs):
        # obtain features
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            outputs = self.extractor(
                **audio_inputs,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = outputs.hidden_states

        all_ones_mask = torch.sum(outputs['attention_mask'], dim=1).tolist()


        selected_features = [hidden_states[i] for i in self.config.target_layers]
        features = torch.stack(selected_features, dim=1).float()

        features = torch.stack([features[idx, :, 29:, :] for idx, i in enumerate(all_ones_mask)])
        # print(features.size())
        # import pdb; pdb.set_trace()
        return self.classifier(features)



# # 8. 主流程
# if __name__ == "__main__":
#     # 1. 配置参数
#     class Config:
#         train_dir = "./ft_datas/fold1.jsonl"
#         val_dir = "./eval_datas/sc_test_data.jsonl"
#         model_name = "/home/papa/data/papa/projects/speechllm/Qwen2-Audio-7B-Instruct"
#         target_layers = [4, 8, 12, 24, 32]  # 选择提取特征的Transformer层
#         seq_pool = "end"
#         select_seq = 10
#         feature_aggregation = "mean"  # 聚合策略: mean/max/concat
#         torch_dtype="bfloat16",
#         num_classes = 4  # 分类类别数
#         batch_size = 8
#         learning_rate = 1e-4
#         epochs = 10
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#     config = Config()
    
#     # 初始化处理器和模型
#     processor = AutoProcessor.from_pretrained(config.model_name)
#     model = QwenAudioClassifier(config)
    
    # 准备数据集 (需替换为实际数据)
    # 示例结构：file_paths = ["audio1.wav", ...], labels = [0, 1, ...]
    # train_files, train_labels, valid_files, valid_labels = prepare_data(config.train_dir, config.val_dir)

    # train_dataset = AudioDataset(train_files, train_labels, processor)
    # valid_dataset = AudioDataset(valid_files, valid_labels, processor)
    
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=0)
    # valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False)
    # import tqdm
    # for batch in tqdm.tqdm(valid_loader):
    #     print(batch[1].size())
    #     # pass
    #     # try:
    #     #     continue
    #     # except:
    #         # pass
    #         # import pdb
    #         # pdb.set_trace()
    # # 开始训练
    # train_model(config, model, train_loader, valid_loader)
    
    # # 保存模型
    # torch.save(model.classifier.state_dict(), "qwen_audio_classifier.pth")