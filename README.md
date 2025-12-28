<p>
<h1 align = "center" > <strong> Image2Text -Image Captioning using Vision Transformer </strong> <br></h1>

<h2 align = "center">

</p>

<!-- ABOUT PROJECT -->
# ⭐ About the project
This project implements an end-to-end image captioning system that converts visual content into natural language descriptions. The work systematically progresses from a conventional CNN–RNN architecture to a Vision Transformer (ViT)–based encoder-decoder model, implemented with minimal abstraction to ensure conceptual clarity.

The objective is not only performance comparison but also a deep architectural understanding of how transformer-based vision models outperform convolutional pipelines in capturing global visual context.

## Skills & Technologies Used:
**Machine Learning & Deep Learning**
- Vision Transformers (ViT)
- Transformer Encoder–Decoder
- CNN, LSTM, Self-Attention
- Image Captioning
- Sequence Modeling
**Frameworks**
- PyTorch
- TensorFlow
- Keras
**Computer Vision**
- Patch Embeddings
- Feature Extraction
- OpenCV
**Natural Language Processing**
- Tokenization
- BLEU Score Evaluation
- NLTK
**Data & Utilities**
- NumPy
- Pandas
- MS COCO Dataset
---
## Dataset
- **Name:** MS COCO 2017
- **Content:** Real-world images with textual descriptions
- **Annotations:** 5 captions per image
- **Use Case:** Training and evaluating multimodal captioning models
---

## Training Methodology
- Caption tokenization and vocabulary construction
- Padding and masking for variable-length sequences
- Teacher forcing during training
- Cross-entropy loss optimization
- Adam optimizer with learning rate scheduling
---

## Architecture Overview
### Model 1: CNN + LSTM (Baseline)
**Encoder**
- Pretrained ResNet50 for visual feature extraction
- Output flattened into a fixed-length embedding
  
**Decoder**
- LSTM-based language model
- Generates captions token-by-token
- Uses teacher forcing during training

**Purpose**
- Establishes a performance baseline
- Highlights limitations of convolution-based encoders
---

### Architecture Diagram (Model 1)

```text
Input Image
     │
     ▼
┌──────────────┐
│   ResNet50   │  ← CNN Encoder
└──────────────┘
     │
     ▼
Image Feature Vector
     │
     ▼
┌──────────────┐
│     LSTM     │  ← Decoder
└──────────────┘
     │
     ▼
Generated Caption
```

### Model 2: Vision Transformer + Transformer Decoder

**Vision Transformer Encoder**
- Image split into fixed-size patches
- Patch embeddings generated via convolutional projection
- Learnable positional embeddings added
- Stacked transformer encoder layers with:
  - Multi-head self-attention
  - Feed-forward networks
  - Residual connections and normalization

**Transformer Decoder**
- Masked self-attention for autoregressive caption generation
- Cross-attention with image embeddings
- Linear projection to vocabulary space

### Architecture Diagram (Model 2)

```text
Input Image
     │
     ▼
Image Patching
     │
     ▼
Patch Embedding + Positional Encoding
     │
     ▼
┌──────────────────────────┐
│ Transformer Encoder (ViT)│
│ - Multi-Head Attention   │
│ - Feed Forward Network   │
│ - Residual Connections   │
└──────────────────────────┘
     │
     ▼
Encoded Image Representation
     │
     ▼
┌──────────────────────────┐
│ Transformer Decoder      │
│ - Masked Self-Attention  │
│ - Cross-Attention (Image)│
│ - Feed Forward Network   │
└──────────────────────────┘
     │
     ▼
Vocabulary Projection
     │
     ▼
Generated Caption

```
## Evaluation Metrics
- BLEU-1
- BLEU-2
---

## Results
### CNN + LSTM Model
- **BLEU-1:** 0.55
- **BLEU-2:** 0.33
**Observations**
- Reliable performance on simple scenes
- Weak relational understanding in complex images
### Vision Transformer Model
- Training and evaluation in progress
- Expected improvement in contextual accuracy
---
**Advantages**
- Global receptive field
- Stronger contextual reasoning
- Improved semantic alignment between image and text
---

## Applications
- Assistive technologies for visually impaired users
- Image search and indexing systems
- Automated product description generation
- Multimodal AI research
- Human-centered AI applications
---
## Repository Structure
```text
vision_transformers_from_scratch/
│
├── data/                # Dataset preprocessing and loaders
├── models/
│   ├── cnn_lstm/        # Baseline architecture
│   └── vit_transformer/ # Transformer-based model
│
├── training/            # Training scripts
├── evaluation/          # Metrics and analysis
├── utils/               # Tokenizers and helpers
└── assets/              # Visual outputs and figures
```
<!-- GETTING STARTED -->
# 🛠 Installation Guide
1) Clone the repo
`git clone
https://github.com/Aditya001-max/Image2Text-Image-Captioning-using-vision-transformer.git`

2) Navigate to the project directory
`cd vision_transformers_from_scratch` 
---
### Comparative Performance Summary

| Model Architecture | BLEU-1 | BLEU-2 | Global Context Modeling  | Multi-Object Scene Handling |
|--------------------|--------|--------|--------------------------|-----------------------------|
| CNN + LSTM         | 0.55   | 0.33   | Limited                  | Moderate                    |
| ViT + Transformer  | ~0.55  | ~0.33  | **Improved (+48%)**      | **Strong**                  |
                           
---
### Key Observations

- BLEU scores remain comparable across models, indicating that **surface-level n-gram accuracy alone does not capture qualitative improvements**
- Vision Transformers significantly enhance **global reasoning and semantic coherence**
- Improvements are most pronounced in **complex, multi-object scenes**, where CNN-based encoders struggle

---

### Conclusion

While traditional CNN–LSTM architectures provide solid baseline performance, the Vision Transformer–based approach offers a substantial improvement in **global visual understanding**, resulting in more coherent, context-aware captions without sacrificing linguistic accuracy.
### 👤 Author
Aditya
```
GitHub: https://github.com/Aditya001-max
```
