# PresentLoop — Real-time Engagement Detection

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-FF6B35?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-4299E1?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**PresentLoop** is a privacy-preserving ambient presence system designed for elderly care and dementia support. It uses MediaPipe pose estimation and machine learning to detect real-time engagement levels from webcam feeds or uploaded videos, without capturing identifiable facial features.

## ✨ Features

- **Live Webcam Detection**: Real-time pose tracking and engagement scoring
- **Batch Video Analysis**: Process multiple MP4/AVI videos automatically
- **Multiple ML Models**: Compare 4 trained RandomForest models (5-20 features)
  - Model 1: Original 5-feature baseline
  - Model 2: Enhanced 11-feature model  
  - Model 3: Probability-averaging ensemble of all models
  - RF Model: 20-feature expanded model
- **Visual Analytics**: Smoothed engagement graphs + comparison plots (PNG exports)
- **Evaluation Metrics**: Auto-saves results to `evaluation_results.csv`
- **Threshold Controls**: Adjustable engagement confidence (0.0-1.0)

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

## 🏗️ Architecture

```
app.py (Streamlit UI + Core Logic)
├── MediaPipe Pose (33 landmarks → key points)
├── extract_features() → Model-specific (5/11/20 feats)
│   ├── Head orientation (sin yaw/pitch)
│   ├── Body pose (shoulder/hip angles)
│   ├── Movement variance
│   └── Visibility confidence
├── Models (*.pkl):
│   ├── presentloop_model.pkl (5 feats)
│   ├── presentloop_model2.pkl (11 feats)  
│   ├── presentloop_model3.pkl (Ensemble)
│   └── rf_model.pkl (20 feats)
├── process_video_multi() → Frame-by-frame scoring
├── generate_engagement_graph() → Smoothed plots
└── evaluation_results.csv → Aggregated metrics
```

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

## 🛠️ Model Development

```bash
# Inspect models
python inspect_models.py

# Recreate ensemble (Model 3)
python create_model3.py
```

**Feature Evolution**:
- **Model 1** (5 feats): Head angles + basic movement
- **Model 2** (11 feats): + Head-shoulder distance, motion variance, pose confidence  
- **RF Model** (20 feats): + Limb positions/visibility
- **Model 3**: Proba-average ensemble (robust to feature mismatches)

## 📂 Project Structure

```
├── app.py                 # Main Streamlit app
├── create_model3.py       # Ensemble creation
├── inspect_models.py      # Model diagnostics
├── *.pkl                  # Trained models
├── pose_landmarker_lite.task  # MediaPipe model
├── evaluation_results.csv # Results log
├── engagement_*.png       # Generated graphs
├── comparison_*.png       # Model comparison plots
└── TODO.md              # Development roadmap
```

## 🔬 Research Context

Built for **elderly engagement monitoring**:
- Detects attentive posture vs distraction/disengagement
- No facial recognition → Privacy-first
- Ambient operation (no user interaction needed)
- Calibrated threshold (default 0.5)

## 🤝 Contributing

1. Add new pose features to `extract_features()`
2. Train models on labeled engagement data
3. Extend UI (e.g., export video overlays)
4. See [TODO.md](TODO.md)

## 📄 License

MIT License — Research prototype for non-commercial use.

---

**Built with ❤️ for accessible AI monitoring** | [Streamlit App](https://streamlit.io) | [MediaPipe Pose](https://mediapipe.dev)

