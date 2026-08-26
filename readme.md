# 🔌 ProbPlug

Official implementation of **ProbPlug**, a plug-and-play module for uncertainty quantification in large-model classification.

ProbPlug attaches a lightweight attention-based classifier to a frozen large model. It aggregates hidden representations from multiple layers and produces probabilistic predictions without modifying or fine-tuning the backbone. The current release supports binary speech-emotion classification based on Qwen2-Audio.

## 🔍 Overview

```text
Audio → Frozen Qwen2-Audio → Multi-layer Hidden States
      → Attention-based ProbPlug → Prediction Probability
```

## 🛠️ Environment

```bash
git clone https://github.com/ChuhangLiu2002/ProbPlug.git
cd ProbPlug

conda create -n probplug python=3.10 -y
conda activate probplug

# Install the PyTorch version compatible with your CUDA environment.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r Speech/requirements.txt
```

The current implementation requires an NVIDIA GPU and supports loading Qwen2-Audio in 8-bit or 4-bit mode.

## 📦 Data Preparation

Prepare the metadata as JSONL files. For training and validation:

```json
{"audios": ["/path/to/audio.wav"], "response": "Angry"}
```

For evaluation:

```json
{"wav": "/path/to/audio.wav", "emo": "ang"}
```

The current binary task maps `Angry`/`ang` to class 1 and all other supported emotions to class 0. Audio is resampled to 16 kHz automatically.

## 🚀 Training

Set the dataset paths, Qwen2-Audio checkpoint, and output directory in the `Config` class of `Speech/binary_trainer.py`, then run:

```bash
cd Speech
python binary_trainer.py
```

Checkpoints and TensorBoard logs will be saved under `save_dir`.

## 📊 Evaluation

Set `test_dir`, `model_name`, `save_dir`, and `resume` in `Speech/binary_eval.py`, then run:

```bash
cd Speech
python binary_eval.py
```

The evaluation script reports WA, UA, micro-F1, macro-F1, and the confusion matrix. It also exports the positive-class probability for each sample.

> 💡 **Note:** `Speech/configs/config.yaml` is provided as an example. The current training and evaluation scripts use their internal `Config` classes.


## 📖 Citation

The paper and BibTeX entry will be released soon. For now, please cite this repository:

```text
https://github.com/ChuhangLiu2002/ProbPlug
```

## 🙏 Acknowledgements

This project is built upon [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio), [Transformers](https://github.com/huggingface/transformers), and [PyTorch](https://pytorch.org/).
