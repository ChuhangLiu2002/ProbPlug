import json
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import librosa
import random
import numpy as np


def handle_json2index(json_file):
    train_data = []
    emotion_idx = {"Happy": 0, "Sad": 1, "Angry": 2, "Neutral": 3, "hap": 0, "sad": 1,  "exc":0, "ang": 2, "neu": 3}
    with open(json_file) as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    try:
        train_files = [i['audios'] for i in train_data]
        train_labels = [emotion_idx[i['response']] for i in train_data]
    except:
        train_files = [i['wav'] for i in train_data]
        train_labels = [emotion_idx[i['emo']] for i in train_data]
    # import pdb
    # pdb.set_trace()
    return train_files, train_labels

def handle_binary_json2index(json_file):
    train_data = []
    emotion_idx = {"Happy": 0, "Sad": 0, "Angry": 1, "Neutral": 0, "hap": 0, "sad": 0,  "exc":0, "ang": 1, "neu": 0}
    with open(json_file) as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    try:
        train_files = [i['audios'] for i in train_data]
        train_labels = [emotion_idx[i['response']] for i in train_data]
    except:
        train_files = [i['wav'] for i in train_data]
        train_labels = [emotion_idx[i['emo']] for i in train_data]
    return train_files, train_labels

def prepare_data(train_data_info_file, val_data_info_file):
    train_files, train_labels = handle_json2index(train_data_info_file)
    val_files, val_labels = handle_json2index(val_data_info_file)
    return train_files, train_labels, val_files, val_labels


def prepare_binary_data(train_data_info_file, val_data_info_file):
    train_files, train_labels = handle_binary_json2index(train_data_info_file)
    val_files, val_labels = handle_binary_json2index(val_data_info_file)
    return train_files, train_labels, val_files, val_labels


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 加载音频并处理 [6,7](@ref)
        audio, sr = librosa.load(self.file_paths[idx][0], sr=16000)
        conversation = [
        {'role': 'system', 'content': '你是一个专业的情感分析大师'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio},
            {"type": "text", "text": "请识别这段语音的情绪<audio>"}
        ]}
    ] 
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # print(text)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", truncation=True, sampling_rate=16000, padding=False)
        # inputs.input_ids = inputs.input_ids.to("cuda")
        return inputs, torch.tensor(self.labels[idx])


class BinaryEmoAudioDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.file_paths)
    @staticmethod
    def sample_augmentation(X, target_sr):
        if target_sr == 16000:
            return X
        # 降采样至16kHz
        audio_tr = librosa.resample(X, orig_sr=16000, target_sr=target_sr, res_type='zero_order_hold')
        audio_tr = librosa.resample(X, orig_sr=target_sr, target_sr=16000)
        return audio_tr
    def __getitem__(self, idx):
        # 加载音频并处理 [6,7](@ref)
        random.seed(42)
        audio, sr = librosa.load(self.file_paths[idx][0], sr=16000)

        # concandiate_srs = [8000, 8000, 8000, 6000, 16000]
        # random_sr = random.choice(concandiate_srs)
        # audio=self.sample_augmentation(audio, random_sr) 

        concandiate_mask = random.uniform(-0.1, 0.3)
        if concandiate_mask < 0:
            audio = np.pad(audio, (concandiate_mask*audio.shape[0], 0), mode='constant')
        else:
            audio[:int(concandiate_mask*audio.shape[0])] = 0

        conversation = [
        {'role': 'system', 'content': '你是一个专业的情感分析大师'}, 
        {"role": "user", "content": [
            {"type": "text", "text": "请识别这段语音是否明确表达了愤怒情绪"},
            {"type": "audio", "audio_url": audio},
        ]}
    ] 
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # print(text)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", truncation=True, sampling_rate=16000,         padding=False)
        # import pdb
        # pdb.set_trace()
        
        # inputs.input_ids = inputs.input_ids.to("cuda")
        return inputs, torch.tensor(self.labels[idx]) # , text


class AudioT_BinaryEmoAudioDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.file_paths)
    @staticmethod
    def sample_augmentation(X, target_sr):
        if target_sr == 16000:
            return X
        # 降采样至16kHz
        audio_tr = librosa.resample(X, orig_sr=16000, target_sr=target_sr, res_type='zero_order_hold')
        audio_tr = librosa.resample(X, orig_sr=target_sr, target_sr=16000)
        return audio_tr
    def __getitem__(self, idx):
        # 加载音频并处理 [6,7](@ref)
        random.seed(42)
        audio, sr = librosa.load(self.file_paths[idx][0], sr=16000)

        # concandiate_srs = [8000, 8000, 8000, 6000, 16000]
        # random_sr = random.choice(concandiate_srs)
        # audio=self.sample_augmentation(audio, random_sr) 

        concandiate_mask = random.uniform(-0.1, 0.3)
        if concandiate_mask < 0:
            audio = np.pad(audio, (concandiate_mask*audio.shape[0], 0), mode='constant')
        else:
            audio[:int(concandiate_mask*audio.shape[0])] = 0

        conversation = [
        {'role': 'system', 'content': '你是一个专业的情感分析大师'}, 
        {"role": "user", "content": [
            {"type": "text", "text": "请识别这段语音是否明确表达了愤怒情绪"},
            {"type": "audio", "audio_url": audio},
        ]}
    ] 
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", truncation=True, sampling_rate=16000,         padding=False)

        # index = torch.nonzero(inputs["input_ids"]==151646)
        
        # import pdb
        # pdb.set_trace()
        
        # inputs.input_ids = inputs.input_ids.to("cuda")
        return inputs, torch.tensor(self.labels[idx]), text



class AudioEvalDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 加载音频并处理 [6,7](@ref)
        audio, sr = librosa.load(self.file_paths[idx], sr=16000)
        conversation = [
        {'role': 'system', 'content': '你是一个专业的情感分析大师'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio},
            {"type": "text", "text": "请识别这段语音的情绪<audio>"}
        ]}
    ] 
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # print(text)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", truncation=True, sampling_rate=16000, padding=False)
        # inputs.input_ids = inputs.input_ids.to("cuda")
        # import pdb
        # pdb.set_trace()
        batch_inputs = {
        "input_ids": inputs['input_ids'].squeeze(0),
        "attention_mask": inputs['attention_mask'].squeeze(0),
        "input_features": inputs['input_features'].squeeze(0),
        "feature_attention_mask": inputs['feature_attention_mask'].squeeze(0),
        }
        return batch_inputs, torch.tensor(self.labels[idx]), self.file_paths[idx]



class BinaryAudioEvalDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 加载音频并处理 [6,7](@ref)
        audio, sr = librosa.load(self.file_paths[idx], sr=16000)
        conversation = [
        {'role': 'system', 'content': '你是一个专业的情感分析大师'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio},
            {"type": "text", "text": "请识别这段语音是否明确表达了愤怒情绪<audio>"}
        ]}
    ] 
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # print(text)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", truncation=True, sampling_rate=16000, padding=False)
        # inputs.input_ids = inputs.input_ids.to("cuda")
        # import pdb
        # pdb.set_trace()
        batch_inputs = {
        "input_ids": inputs['input_ids'].squeeze(0),
        "attention_mask": inputs['attention_mask'].squeeze(0),
        "input_features": inputs['input_features'].squeeze(0),
        "feature_attention_mask": inputs['feature_attention_mask'].squeeze(0),
        }
        return batch_inputs, torch.tensor(self.labels[idx]), self.file_paths[idx]




def collate_fn(batch):
    # 分离输入和标签
    inputs_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    
    # 动态填充同一批次内的input_ids
    input_ids = [x["input_ids"].squeeze(0) for x in inputs_list]  # 移除冗余批次维度
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # 填充到批次最大长度
    
    # 为填充后的input_ids生成attention_mask
    attention_mask = (input_ids != 0).int()

    # 重构批次数据（假设音频特征无需填充）
    batch_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_features": torch.stack([x["input_features"].squeeze(0) for x in inputs_list]),
        "feature_attention_mask": torch.stack([x["feature_attention_mask"].squeeze(0) for x in inputs_list])
    }
    return batch_inputs, labels



def collate_fn(batch):
    # 分离输入和标签
    inputs_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    
    # 动态填充同一批次内的input_ids
    input_ids = [x["input_ids"].squeeze(0) for x in inputs_list]  # 移除冗余批次维度
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # 填充到批次最大长度
    
    # 为填充后的input_ids生成attention_mask
    attention_mask = (input_ids != 0).int()

    # 重构批次数据（假设音频特征无需填充）
    batch_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_features": torch.stack([x["input_features"].squeeze(0) for x in inputs_list]),
        "feature_attention_mask": torch.stack([x["feature_attention_mask"].squeeze(0) for x in inputs_list])
    }
    return batch_inputs, labels
