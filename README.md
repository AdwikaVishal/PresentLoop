# PresentLoop — Real-time Engagement Detection

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-FF6B35?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-4299E1?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**PresentLoop** is an AI-based system designed to detect cognitive engagement in elderly individuals using pose-based behavioral signals. Unlike traditional systems that rely purely on physical activity or facial recognition, this project explores whether movement alone is a reliable proxy for engagement.
The system processes video input (webcam or uploaded), extracts pose features using MediaPipe, and applies machine learning models to classify engagement in real time—while maintaining a privacy-preserving, non-invasive approach.

## 🎯 Problem Statement
Existing engagement detection systems often assume:
- Movement = Engagement
- Stillness = Disengagement

However, in real-world scenarios:
- Walking does not imply attention
- Sitting still may indicate focus
- Sleeping or confusion can be misclassified as engagement

This project aims to address this gap by building a data-driven, context-aware engagement detection system.

## ⚙️ System Architecture
Video Input → Pose Extraction → Feature Engineering → Model Prediction → Temporal Smoothing → Output

**Key Components**:
- **Pose Estimation**: MediaPipe (33 body landmarks)
- **Feature Engineering**:
  - Head orientation (yaw, pitch)
  - Posture stability
  - Motion intensity & variance
  - Pose confidence
- **Models**:
  - Random Forest (5, 11, 20 feature variants)
  - Ensemble model (probability averaging)
- **Interface**: Streamlit-based real-time dashboard

## 🚀 Features
- Real-time engagement detection from webcam
- Batch video analysis with automated evaluation
- Multiple ML models for comparative analysis
- Smoothed engagement visualization over time
- Exportable evaluation metrics (evaluation_results.csv)
- Adjustable confidence threshold for classification

## 📱 Demo

```
streamlit run app.py
```

**Live Mode**:
- Mirror webcam with pose landmarks overlaid
- Real-time engagement score + status (Engaged/Not Engaged)
- Rolling graph of last 100 frames

**Batch Mode**:
1. Upload videos via sidebar
2. Auto-generates per-model graphs (`engagement_*.png`)
3. Comparison plots (`comparison_*.png`)
4. Metrics table appended to CSV

![Screenshot](engagement_Screen%20Recording%202026-04-14%20at%205.50.23%20PM.png)
*Example: Model comparison graphs*



## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install streamlit opencv-python mediapipe scikit-learn pandas matplotlib
   ```

2. **Download Models** (included):
   - `presentloop_model*.pkl`
   - `rf_model.pkl` 
   - `pose_landmarker_lite.task`

3. **Run**:
   ```bash
   streamlit run app.py
   ```

4. **Test Videos**:
   - Upload sample MP4s (temp files auto-cleaned)
   - View `engagement_*.png` + `evaluation_results.csv`

## 📊 Sample Results

From `evaluation_results.csv`:

| Video Name | Model | Avg Score | Engaged % | Classification |
|------------|-------|-----------|-----------|----------------|
| test1 | Model 2 | 0.623 | 68.4% | Engaged |
| test2 | Model 3 | 0.589 | 62.1% | Engaged |

**Model 2 consistently outperforms Model 1** (higher engagement % across test videos).

## 🛠️ Methodology

1. **Baseline Model**
   A rule-based system using posture and motion features:
   - Performed well in static scenarios
   - Failed in dynamic and low-motion cases

2. **Machine Learning Approach**
   Trained Random Forest models on engineered pose features
   Applied:
   - Cross-validation
   - Class balancing
   - Feature expansion (5 → 20 features)

3. **Temporal Modeling**
   Experimented with LSTM networks
   Observed limited performance due to small sequential dataset

**👉 Final insight**: Simpler models outperformed complex ones given the data constraints

## 📱 Demo
```
streamlit run app.py
```

**Live Mode**:
- Real-time pose tracking
- Engagement score visualization
- Rolling graph of predictions

**Batch Mode**:
- Upload videos
- Generate engagement graphs
- Export performance metrics

## 📊 Results
Random Forest achieved ~95% accuracy
Ensemble model improved stability and reduced noise
Smoothed predictions enhanced interpretability

However, evaluation revealed critical real-world limitations (see below).

## 🧠 Key Insights
- Movement ≠ Cognitive Engagement
- Low motion ≠ Disengagement (or engagement)
- Feature engineering had a greater impact than model complexity
- Simpler models can outperform deep learning on limited data
- Evaluation metrics alone can be misleading without contextual analysis

## ⚠️ Limitations
- Pose-based features lack contextual understanding (e.g., gaze, attention)
- Misclassification in low-motion states (e.g., sleeping)
- Sensitivity to camera distance and orientation
- Limited dataset size affects temporal modeling

## 🔮 Future Work
- Integrate multimodal signals (gaze tracking, audio, facial cues)
- Improve temporal modeling using larger sequential datasets
- Introduce context-aware engagement classification
- Expand dataset diversity for real-world robustness

## 📂 Project Structure
```
├── app.py
├── create_model3.py
├── inspect_models.py
├── models/*.pkl
├── pose_landmarker_lite.task
├── evaluation_results.csv
├── engagement_*.png
├── comparison_*.png
└── README.md
```

## 📊 Sample Output
| Video | Model | Avg Score | Engaged % |
|-------|-------|-----------|-----------|
| test1 | Model 2 | 0.62 | 68% |
| test2 | Model 3 | 0.58 | 62% |

## 🔬 Research Contribution
This project highlights a key challenge in human-centered AI:
Physical signals alone are insufficient to infer cognitive states.

By demonstrating the limitations of pose-based engagement detection, PresentLoop contributes toward building more context-aware and reliable assistive AI systems.

## 🤝 Contributing
- Extend feature extraction
- Train models on larger datasets
- Improve UI/UX
- Add multimodal capabilities

## 📄 License

MIT License — Research prototype for non-commercial use.

---

**❤️ Acknowledgment**  
Built with a focus on accessible, privacy-preserving AI for healthcare and assistive technology.

## 📄 License
MIT License — Research prototype for non-commercial use.

