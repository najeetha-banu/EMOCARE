# EMOCARE: AI-Powered Mental Health Monitoring System 🧠❤️

## Overview

A comprehensive deep learning system that monitors mental health by analyzing **facial emotions, text sentiment, behavioral patterns, and fairness metrics**. This capstone project combines computer vision, NLP, graphical models, and ethical AI principles.

## Problem Statement

Traditional mental health monitoring relies on manual assessment and delayed intervention. This project automates emotion detection and combines multiple signals (facial, text, behavioral) for early intervention.

## System Architecture
Input (Image/Video/Text)
↓
├─ Facial Emotion Recognition
│  └─ EfficientNetB0 + Attention (7 emotions)
├─ Text Sentiment Analysis
│  └─ TextBlob + NLP
└─ Behavioral Assessment
└─ Questionnaire + Bayesian Network
↓
Combined Mental Health Signal
↓
Fairness Analysis (Demographic Parity, Equalized Odds)
↓
Final Mental Health Report
## Key Features

### 1. Facial Emotion Recognition
- **Model:** EfficientNetB0 with custom attention mechanism
- **Dataset:** FER2013 (13,000+ images)
- **Classes:** 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Accuracy:** 85% on CK+ test set
- **Input:** Images (48×48 pixels) or video frames

### 2. Multi-Modal Analysis
- **Facial emotions** from images/videos
- **Text sentiment** analysis using TextBlob
- **Behavioral assessment** via questionnaire
- **Combined prediction** merging all signals

### 3. Advanced Features
- **Bayesian Network:** Models emotion relationships using pgmpy
- **Fairness Analysis:** Demographic parity, equalized odds using Fairlearn
- **Video Processing:** Frame-by-frame emotion tracking in GIF/videos
- **Real-time inference:** Supports video streams

### 4. Model Comparison
- **EfficientNetB0** (selected) vs **MobileNet**
- EfficientNet provides better accuracy with efficient computation

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow, Keras |
| Models | EfficientNetB0, MobileNet, Attention mechanism |
| Graphical Models | pgmpy (Bayesian Networks) |
| Fairness | Fairlearn (demographic parity, equalized odds) |
| Text Analysis | TextBlob, NLTK |
| Image Processing | OpenCV, PIL |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## Project Files

| Notebook | Purpose |
|----------|---------|
| `training_with_fer2013.ipynb` | Main model training on FER2013 |
| `testing_with_CK+_dataset.ipynb` | Testing generalization on CK+ |
| `mobilenetnet_comparison.ipynb` | EfficientNet vs MobileNet comparison |
| `capstone_final.ipynb` | Complete integrated system |
| `capstone_test_3_fairness.ipynb` | Fairness analysis & bias mitigation |
| `Capstone_final_test_2.ipynb` | Final testing with fairness metrics |
| `capstonetest.ipynb` | Image + text sentiment combined |

## Results

### Model Performance
- **FER2013 Training Accuracy:** 84.66%
- **CK+ Test Accuracy:** 85.33%
- **Final Integrated System:** 72.77% (realistic end-to-end)

### Fairness Metrics
- **Demographic Parity:** 0.0376-0.0555
- **Equalized Odds:** Analyzed per emotion class
- **Bias Mitigation:** ExponentiatedGradient with DemographicParity

### Key Insights
- Model generalizes well from FER2013 to CK+ dataset
- EfficientNet outperforms MobileNet for emotion classification
- Video-based emotion tracking enables real-time monitoring
- Fairness analysis identifies and mitigates demographic biases

## How to Use

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training
```bash
# Train main model on FER2013
jupyter notebook training_with_fer2013.ipynb
```

### Test on Video/Image
```bash
# Test on GIF/video
jupyter notebook capstone_final.ipynb
```

### Fairness Analysis
```bash
# Check fairness metrics
jupyter notebook capstone_test_3_fairness.ipynb
```

## Dataset Information

| Dataset | Size | Purpose |
|---------|------|---------|
| FER2013 | 13,000+ images | Training |
| CK+ | ~920 images | Testing generalization |

Note: Datasets can be downloaded from:
- FER2013: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- CK+: Available upon request from authors

## Key Learnings

✅ Transfer learning with EfficientNetB0
✅ Custom attention mechanisms for neural networks
✅ Multi-modal machine learning (vision + text + behavioral)
✅ Fairness in machine learning (Fairlearn framework)
✅ Bayesian Networks for probabilistic reasoning
✅ Video/GIF processing with deep learning
✅ Handling imbalanced emotion datasets
✅ Model comparison and selection

## Interview Explanation

> "My capstone project is an AI-powered mental health monitoring system. I trained an EfficientNetB0 model with a custom attention mechanism on FER2013 for 7-emotion facial recognition. I compared it with MobileNet and EfficientNet performed better (85% accuracy on CK+). The system extends beyond images—it processes videos frame-by-frame for temporal emotion tracking. I also integrated text sentiment analysis using TextBlob and behavioral questionnaires. As an ML practitioner, I included fairness analysis using Fairlearn to check for demographic biases. I modeled emotion relationships using a Bayesian Network with pgmpy. The final integrated system achieves 72.77% accuracy in real-world testing scenarios."

## Future Work

- [ ] Real-time webcam emotion streaming
- [ ] Mobile app deployment
- [ ] Privacy-preserving federated learning
- [ ] Multi-language text support
- [ ] Emotion intensity prediction (0-100)
- [ ] Clinical validation with mental health professionals

## Ethical Considerations

This system is designed for **supportive monitoring only**, not diagnosis. It should:
- Complement professional mental health care
- Respect user privacy (on-device processing when possible)
- Account for cultural differences in emotion expression
- Mitigate demographic biases (addressed in fairness analysis)

## Author

**Najeetha Banu**
- M.Tech in AI & Decision Sciences, Manipal Institute of Technology
- First-author IEEE Access publication on CTR prediction and auction simulation
- 10-month internship at SABIC (Data Science)

## Contact

📧 Email: najeetha2002@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/najeetha-banu-37314820a/
🐙 GitHub: https://github.com/najeetha-banu



---

*Project completed: June 2025*
*For questions or collaborations, feel free to reach out!*
