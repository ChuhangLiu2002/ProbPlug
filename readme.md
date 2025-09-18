# Introduction

Large Language Models (LLMs), while performing impressively across many tasks, often exhibit the problem of overconfidence in their outputs. This is especially evident when dealing with domain-specific or data-limited downstream tasks. Such “confidently wrong predictions” can pose potential risks. Therefore, performing Uncertainty Quantification (UQ) on LLM outputs—that is, evaluating the model’s confidence in its results—is crucial for improving their reliability and trustworthiness.
ProbPlug is designed to address this issue as a pluggable adapter when large models are applied to classification tasks. Its core goal is to provide probabilistic uncertainty quantification capabilities for LLMs, specifically in the context of binary classification tasks.


## Fundamental Principle

The fundamental idea of ProbPlug is to attach a pluggable adapter to an LLM. This adapter does not require modifying the original architecture of the LLM, yet it enables uncertainty estimation for its outputs (specifically for binary classification results). Concretely, an attention network is employed, which takes the outputs of each layer of the large model as inputs, and then produces classification probabilities. This module requires additional training.

1. Probabilistic Output: Unlike traditional LLMs that only provide a deterministic class label (e.g., “Yes” or “No”), ProbPlug is designed to allow the model to output a probability distribution or confidence score for its decision. For example, instead of simply predicting “Yes,” the model may also output a probability such as 0.85, indicating that it is 85% confident the result belongs to the “Yes” class.

2. Uncertainty Quantification: With probabilistic outputs, we can better understand the cognitive state of the model. A higher probability value generally indicates that the model is more confident in its judgment, whereas a probability value close to 0.5 (the level of random guessing in binary classification) suggests that the model is uncertain or confused about the input. This helps to identify ambiguous or challenging samples where the model may be prone to errors. At the same time, such measures enable us to make more informed decisions based on the results.


# How to use
### 1.Clone the repository

```bash
git clone git@github.com:ChuhangLiu2002/ProbPlug.git
cd Speech
```

### 2.Set up evironments

```bash
conda create -n probplug python=3.10
conda activate probplug
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 # UPDATE ME!
cd Speech
pip install -r requirements.txt
```

### 3.Train and test

Train

```bash
python binary_trainer.py
```

Test

```bash
python binary_eval.py
```

The code for the text classifier wiil be available soon






























# TODO


# Citition
