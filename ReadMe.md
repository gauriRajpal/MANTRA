# README: Bidirectional Neural Machine Translation for Indic Languages

MANTRA ( Multilingual Adaptive Neural Transaltion Architecture ) implements a **Bidirectional Neural Machine Translation (NMT)** system specifically designed for low-resource Indic languages. It focuses on five primary languages: **Hindi, Bengali, Bhojpuri, Punjabi, and Sindhi**. By utilizing a high-quality curated parallel corpus and the **Transformer architecture**, the system aims to improve translation quality in data-scarce conditions.

---

## 🚀 Key Features

* **Multilingual Support**: Enables translation between Hindi, Bengali, Bhojpuri, Punjabi, and Sindhi.
* **Engine Variety**: Features 12 specific bidirectional models and a unified multilingual model leveraging cross-lingual transfer learning.
* **Transformer Architecture**: Built with 6 encoder layers, 6 decoder layers, and 8 attention heads.
* **Sub-word Tokenization**: Employs **SentencePiece** (unigram) to handle morphological richness and the "Out-of-Vocabulary" (OOV) problem.
* **Interactive GUI**: Includes a dark-themed user interface developed with **Streamlit** for real-time inference.

---

## 🛠️ Project Pipeline

1.  **Data Collection**: Parallel corpora (~50,000 pairs per language) sourced from reliable news outlets like BBC Hindi and Dainik Bhaskar.
2.  **Data Filtering**: Rigorous cleaning to remove duplicates, misaligned sentences, and extreme length variations.
3.  **Preprocessing**: Text normalization and sub-word units creation to enhance model efficiency.
4.  **Training**: Executed using the **OpenNMT** framework on **NVIDIA A100** GPUs with FP16 precision.
5.  **Evaluation**: Primary performance measurement via **BLEU scores** on both seen and unseen datasets.

---

## 🛠️ Methodology

The project follows a structured data-centric pipeline to ensure high-fidelity translations despite limited resources.

### 1. Corpus Collection & Alignment
* **Source Data**: Hindi sentences were collected from reliable news sources including Dainik Bhaskar, Dainik Jagran, BBC Hindi, and Aaj Tak.
* **Domain Diversity**: The data covers multiple domains such as Health, Finance, Sports, Entertainment, and Tourism to ensure a diverse vocabulary.
* **Multilingual Expansion**: Collected Hindi sentences were translated into Bengali, Bhojpuri, Punjabi, and Sindhi.
* **Alignment Strategy**: Data is organized in the "Moses" format, maintaining a strict line-by-line semantic alignment across all language files.
* **Dataset Size**: Approximately 50,000 sentence pairs were prepared for each language.

### 2. Data Filtering Pipeline
A rigorous cleaning pipeline was implemented to improve model convergence and performance:
* **Empty & Duplicate Removal**: Deletion of null values, incomplete segments, and repeated sentence pairs.
* **Length Constraints**: Removal of extremely long sentences that strain memory and very short sentences lacking context.
* **Sanitization**: Removal of HTML tags and elimination of misaligned source-target pairs.

### 3. Tokenization & Sub-wording
To handle the "Out-of-Vocabulary" (OOV) problem common in morphologically rich Indic languages, the project employs **SentencePiece**:
* **Sub-word Units**: Words are decomposed into smaller, meaningful units to handle rare tokens effectively.
* **Model Training**: A unigram sub-word model was trained on the filtered datasets and applied to create the sub-worded corpora.
* Models are available in:  https://huggingface.co/GauriRajpal/NMT

### 4. Neural Network Training
The system utilizes the **Transformer architecture** implemented via the **OpenNMT** framework:
* **Model Configuration**: Features 6 encoder layers, 6 decoder layers, and 8 attention heads.
* **Optimization**: Uses the Adam optimizer with "noam" decay and label smoothing (0.1) to prevent overconfidence.
* **Hardware Acceleration**: Training is executed on NVIDIA A100 GPUs using FP16 precision for faster gradient calculations.
* **Learning Paradigms**: Features 12 bidirectional models and a unified multilingual model leveraging cross-lingual transfer learning.

### 5. Evaluation Strategy
* **Metric**: Primary performance is measured using the **BLEU (Bilingual Evaluation Understudy)** score.
* **Dataset Partitioning**: Data is split into Training, Development (for hyperparameter tuning), and Testing (unseen data for unbiased evaluation) subsets.
* **Qualitative Analysis**: Manual line-by-line comparisons were performed on 100 unseen sentences to verify linguistic fluency and adequacy.


---

## 📊 Performance (Sample BLEU Scores)

| Language Pair | BLEU Score |
| :--- | :--- |
| **Hindi-Bhojpuri** | 79.63 |
| **Punjabi-Hindi** | 48.88 |
| **Hindi-Punjabi** | 46.23 |
| **Sindhi-Punjabi** | 32.10 |
| **Bengali-Bhojpuri** | 28.29 |

---

## 💻 Technical Stack

* **Frameworks**: PyTorch, OpenNMT
* **Sub-wording**: SentencePiece
* **Deployment**: Streamlit
* **Hardware**: NVIDIA A100-SXM4-80GB
* **Evaluation**: SacreBLEU

---

## 🔮 Future Work

* **Dataset Scaling**: Increasing the size of parallel corpora via web scraping.
* **Pre-trained Models**: Exploring fine-tuning on **mBART** or **mT5**.
* **Voice Translation**: Extending the system to support voice-based real-time translation.

---

## 🤝 Acknowledgements

Developed at the **Speech and Language Processing Lab, Center for Artificial Intelligence, Banasthali Vidyapith**.

**Supervised by**: Dr. Nisheeth Joshi
**Team Members**: Ananya Kumari, Gauri Rajpal, Harshikha Joshi, Harshita Khanduja
