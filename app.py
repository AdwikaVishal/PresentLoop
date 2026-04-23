import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt

class ProbaAverageEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict_proba(self, X):
        probas = []
        for m in self.models:
            n = m.n_features_in_
            probas.append(m.predict_proba(X[:, :n]))
        return np.mean(probas, axis=0)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    @property
    def n_features_in_(self):
        return max(m.n_features_in_ for m in self.models)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load the trained Random Forest model

def load_models():
    """Load all pre-trained engagement classification models."""
    models = {}
    with open('presentloop_model.pkl', 'rb') as f:
        models['model1'] = pickle.load(f)
    with open('presentloop_model2.pkl', 'rb') as f:
        model2_data = pickle.load(f)
        models['model2'] = model2_data['model']
    with open('presentloop_model3.pkl', 'rb') as f:
        model3_data = pickle.load(f)
        models['model3'] = model3_data['model']
    return models

models = load_models()

import pandas as pd
import os
from pathlib import Path

def process_video_multi(video_path, models):
    """
    Process video file to extract engagement scores per frame for both models.
    Returns {'model1': scores1, 'model2': scores2}
    """
    cap = cv2.VideoCapture(video_path)
    scores1, scores2, scores3 = [], [], []
    
    # Reset prev_features for video processing
    if hasattr(st.session_state, 'prev_features_video'):
        del st.session_state.prev_features_video
    st.session_state.prev_features_video = None
    if hasattr(st.session_state, 'prev_key_landmarks_video'):
        del st.session_state.prev_key_landmarks_video
    st.session_state.prev_key_landmarks_video = None
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Override session_state temporarily for extract_features
            orig_prev_features = st.session_state.prev_features if hasattr(st.session_state, 'prev_features') else None
            orig_prev_key_landmarks = getattr(st.session_state, 'prev_key_landmarks', None)
            
            st.session_state.prev_features = st.session_state.prev_features_video
            st.session_state.prev_key_landmarks = st.session_state.prev_key_landmarks_video
            
            features1 = extract_features(results.pose_landmarks.landmark, models['model1'])
            features2 = extract_features(results.pose_landmarks.landmark, models['model2'])
            features3 = extract_features(results.pose_landmarks.landmark, models['model3'])
            
            # Restore + update video session state
            st.session_state.prev_features_video = st.session_state.prev_features
            st.session_state.prev_key_landmarks_video = st.session_state.prev_key_landmarks
            st.session_state.prev_features = orig_prev_features
            st.session_state.prev_key_landmarks = orig_prev_key_landmarks
            
            # Inline prediction
            features1_2d = features1.reshape(1, -1)
            probas1 = models['model1'].predict_proba(features1_2d)[0]
            score1 = probas1[1]
            
            features2_2d = features2.reshape(1, -1)
            probas2 = models['model2'].predict_proba(features2_2d)[0]
            score2 = probas2[1]
            
            features3_2d = features3.reshape(1, -1)
            probas3 = models['model3'].predict_proba(features3_2d)[0]
            score3 = probas3[1]
            
            scores1.append(score1)
            scores2.append(score2)
            scores3.append(score3)
        else:
            scores1.append(0.0)
            scores2.append(0.0)
            scores3.append(0.0)
        
        frame_count += 1
    
    cap.release()
    return {'model1': scores1, 'model2': scores2, 'model3': scores3}



def process_video(video_path, model_key):
    """Legacy: process single model by key."""
    scores_dict = process_video_multi(video_path, models)
    return scores_dict[model_key]

def generate_engagement_graph(scores, video_name):
    """
    Generate smoothed engagement graph and save as PNG.
    """
    if not scores:
        return None
    
    frames = np.arange(len(scores))
    
    # Moving average smoothing (window=10)
    window = 10
    smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
    smoothed_frames = frames[:len(smoothed)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frames, scores, 'b-', alpha=0.5, label='Raw Engagement', linewidth=0.8)
    ax.plot(smoothed_frames, smoothed, 'r-', linewidth=2.5, label=f'Smoothed (window={window})')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Engagement Score')
    ax.set_title(f'Engagement Analysis: {video_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save PNG
    png_path = f'engagement_{video_name}.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return png_path  # Return path for UI use

def generate_evaluation_table(scores_dict, video_name, model_name):
    """
    Generate evaluation table row for single model or comparison and append to CSV.
    """
    if model_name == "Compare Both":
        scores1 = scores_dict['model1']
        scores2 = scores_dict['model2']
        scores3 = scores_dict.get('model3', [])
        if not scores1:
            return pd.DataFrame()
        
        total_frames = len(scores1)
        rows = []
        for label, scores in [('Model 1', scores1), ('Model 2', scores2), ('Model 3', scores3)]:
            if not scores:
                continue
            avg = np.mean(scores)
            engaged = sum(1 for s in scores if s > 0.5)
            rows.append({
                'Video Name': video_name,
                'Model': label,
                'Average Score': f"{avg:.3f}",
                'Max Score': f"{np.max(scores):.3f}",
                'Min Score': f"{np.min(scores):.3f}",
                'Engaged Frames': engaged,
                'Engaged %': f"{(engaged / total_frames * 100):.1f}",
                'Classification': "Engaged" if avg > 0.5 else "Not Engaged"
            })
        df = pd.DataFrame(rows)
    else:
        key_map = {'Model 1 (Original)': 'model1', 'Model 2 (New)': 'model2', 'Model 3 (Ensemble)': 'model3'}
        key = key_map.get(model_name, 'model1')
        scores = scores_dict.get(key) or list(scores_dict.values())[0]
        if not scores:
            return pd.DataFrame()
        
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        engaged_frames = sum(1 for s in scores if s > 0.5)
        total_frames = len(scores)
        percentage = (engaged_frames / total_frames) * 100 if total_frames > 0 else 0
        classification = "Engaged" if avg_score > 0.5 else "Not Engaged"
        
        table_data = {
            'Video Name': video_name,
            'Model': model_name,
            'Average Score': f"{avg_score:.3f}",
            'Max Score': f"{max_score:.3f}",
            'Min Score': f"{min_score:.3f}",
            'Engaged Frames': engaged_frames,
            'Engaged %': f"{percentage:.1f}",
            'Classification': classification
        }
        df = pd.DataFrame([table_data])
    
    # Append to CSV
    csv_path = 'evaluation_results.csv'
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    
    st.success(f"📋 Table saved/appended to {csv_path}")
    return df

def extract_features(landmarks, model):
    """
    Extract engagement features from MediaPipe pose landmarks for specific model.
    
    model1 (5 features): head_x(sinyaw), head_y(sinpitch), shoulder_width(sibody), distance, movement
    model2 (11 features): above + head_to_shoulder_dist, motion_variance, head_tilt, torso_length,
                          hip_visibility, min_wrist_dist, pose_confidence
    
    Dynamically returns correct shape based on model.n_features_in_
    """
    if len(landmarks) == 0:
        if hasattr(model, 'n_features_in_'):
            return np.zeros(model.n_features_in_)
        return np.zeros(5)
    
    # Key landmarks (MediaPipe 33-pose)
    indices = {
        'nose':0, 'l_eye':2, 'r_eye':5, 'l_ear':7, 'r_ear':8,
        'l_shoulder':11, 'r_shoulder':12, 'l_elbow':13, 'r_elbow':14,
        'l_wrist':15, 'r_wrist':16, 'l_hip':23, 'r_hip':24
    }
    h, w = 640, 480
    
    def get_landmark(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w, lm.visibility])
    
    # Extract positions + visibility
    points = {k: get_landmark(v) for k,v in indices.items()}
    
    # Common features (match model1 names, first 5 for model2)
    l_ear, r_ear = points['l_ear'][:3], points['r_ear'][:3]
    l_eye, r_eye = points['l_eye'][:2], points['r_eye'][:2]
    nose = points['nose'][:2]
    l_shoulder, r_shoulder = points['l_shoulder'][:3], points['r_shoulder'][:3]
    
    sin_yaw = np.sin(np.arctan2(r_ear[0]-l_ear[0], np.linalg.norm(r_ear[:2]-l_ear[:2])))
    eye_center = np.mean([l_eye, r_eye], 0)
    sin_pitch = np.sin(np.arctan2(nose[1]-eye_center[1], np.linalg.norm(nose-eye_center)))
    shoulder_vec = r_shoulder[:2] - l_shoulder[:2]
    sin_body = np.sin(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
    distance = np.tanh(np.mean([l_shoulder[2], r_shoulder[2]]) / 100)
    
    # Movement (model1-style mean delta, update prev_features)
    movement = 0.0
    if st.session_state.prev_features is not None:
        movement = np.mean(np.abs(np.array(st.session_state.prev_features[:4]) - np.array([sin_yaw, sin_pitch, sin_body, distance])))
    st.session_state.prev_features = [sin_yaw, sin_pitch, sin_body, distance, movement]
    
    n_feats = model.n_features_in_ if hasattr(model, 'n_features_in_') else 5
    
    if n_feats == 5:
        return np.array([sin_yaw, sin_pitch, sin_body, distance, movement])
    
    elif n_feats in (11, 20):
        # Additional model2/model3 features
        shoulder_mid = (l_shoulder[:2] + r_shoulder[:2]) / 2
        head_to_shoulder = np.tanh(np.linalg.norm(nose - shoulder_mid) / 100)
        
        curr_key_pos = np.array([p[:2] for p in points.values()])
        motion_var = 0.0
        if st.session_state.prev_key_landmarks is not None:
            deltas = np.abs(curr_key_pos - st.session_state.prev_key_landmarks)
            motion_var = np.var(deltas)
        st.session_state.prev_key_landmarks = curr_key_pos
        
        head_tilt = np.sin(np.arctan2(r_ear[1]-l_ear[1], r_ear[0]-l_ear[0]))
        l_hip = points['l_hip'][:3]
        torso_len = np.tanh(np.linalg.norm(l_shoulder[:2] - l_hip[:2]) / 100)
        hip_vis = np.mean([points['l_hip'][3], points['r_hip'][3]])
        l_wrist_dist = np.linalg.norm(points['l_wrist'][:2] - shoulder_mid)
        r_wrist_dist = np.linalg.norm(points['r_wrist'][:2] - shoulder_mid)
        min_wrist_dist = np.tanh(min(l_wrist_dist, r_wrist_dist) / 100)
        pose_conf = np.mean([p[3] for p in points.values()])
        
        base11 = np.array([
            sin_yaw, sin_pitch, sin_body, head_to_shoulder, movement,
            motion_var, head_tilt, torso_len, hip_vis, min_wrist_dist, pose_conf
        ])
        
        if n_feats == 11:
            return base11
        
        # 9 extra features to reach 20 for rf_model
        r_hip = points['r_hip'][:3]
        hip_width = np.tanh(np.linalg.norm(points['l_hip'][:2] - points['r_hip'][:2]) / 100)
        l_elbow_dist = np.tanh(np.linalg.norm(points['l_elbow'][:2] - shoulder_mid) / 100)
        r_elbow_dist = np.tanh(np.linalg.norm(points['r_elbow'][:2] - shoulder_mid) / 100)
        nose_to_hip = np.tanh(np.linalg.norm(nose - (points['l_hip'][:2] + points['r_hip'][:2]) / 2) / 100)
        shoulder_width = np.tanh(np.linalg.norm(l_shoulder[:2] - r_shoulder[:2]) / 100)
        l_wrist_vis = points['l_wrist'][3]
        r_wrist_vis = points['r_wrist'][3]
        l_elbow_vis = points['l_elbow'][3]
        r_elbow_vis = points['r_elbow'][3]
        
        return np.concatenate([base11, [
            hip_width, l_elbow_dist, r_elbow_dist, nose_to_hip,
            shoulder_width, l_wrist_vis, r_wrist_vis, l_elbow_vis, r_elbow_vis
        ]])
    
    else:
        raise ValueError(f"Unsupported model with {n_feats} features")

# predict_engagement removed: features now extracted with model-specific logic
# Inline in callers for clarity

# Streamlit App Layout
st.set_page_config(page_title="PresentLoop", page_icon="🔄", layout="wide")

st.title("🔄 PresentLoop — Real-time Engagement Detection")
st.markdown("Privacy-preserving ambient presence system for elderly care and dementia support.")

# Sidebar for controls
st.sidebar.header("Controls")
run_live = st.sidebar.checkbox("Run Live Webcam", value=True)
run_uploaded = st.sidebar.checkbox("Process Uploaded Media (Image/Video)", value=False)
confidence_threshold = st.sidebar.slider("Engagement Threshold", 0.0, 1.0, 0.5)

selected_model = st.sidebar.radio(
    "Select Model", 
    ["Model 1 (Original)", "Model 2 (New)", "Model 3 (Ensemble)", "Compare Both"],
    index=0
)

if selected_model == "Model 1 (Original)":
    model_key = 'model1'
elif selected_model == "Model 2 (New)":
    model_key = 'model2'
elif selected_model == "Model 3 (Ensemble)":
    model_key = 'model3'
else:
    model_key = None
current_model_key = model_key
current_model = models[model_key] if model_key else None


# Main columns for layout
col1, col2 = st.columns([2, 1])

# Initialize session state for plotting
if 'engagement_history' not in st.session_state:
    st.session_state.engagement_history = deque(maxlen=100)
if 'engagement_history1' not in st.session_state:
    st.session_state.engagement_history1 = deque(maxlen=100)
if 'engagement_history2' not in st.session_state:
    st.session_state.engagement_history2 = deque(maxlen=100)
if 'engagement_history3' not in st.session_state:
    st.session_state.engagement_history3 = deque(maxlen=100)
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'prev_features' not in st.session_state:
    st.session_state.prev_features = None
if 'prev_key_landmarks' not in st.session_state:
    st.session_state.prev_key_landmarks = None

# Webcam processing
frame_placeholder = col1.empty()
score_placeholder = col2.empty()
status_placeholder = col2.empty()
graph_placeholder = col2.empty()

if run_live:
    # Webcam capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened() and st.session_state.frame_count < 1000:  # Limit for demo
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose detection
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks
        annotated_frame = rgb_frame.copy()
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            if selected_model == "Compare Both":
                features1 = extract_features(results.pose_landmarks.landmark, models['model1'])
                features2 = extract_features(results.pose_landmarks.landmark, models['model2'])
                features3 = extract_features(results.pose_landmarks.landmark, models['model3'])
                
                score1 = models['model1'].predict_proba(features1.reshape(1, -1))[0][1]
                score2 = models['model2'].predict_proba(features2.reshape(1, -1))[0][1]
                score3 = models['model3'].predict_proba(features3.reshape(1, -1))[0][1]
                status1 = "Engaged" if score1 > 0.5 else "Not Engaged"
                status2 = "Engaged" if score2 > 0.5 else "Not Engaged"
                status3 = "Engaged" if score3 > 0.5 else "Not Engaged"
                
                st.session_state.engagement_history1.append(score1)
                st.session_state.engagement_history2.append(score2)
                st.session_state.engagement_history3.append(score3)
                st.session_state.frame_count += 1
                
                col_frame, col_m1, col_m2_graph, col_m3 = st.columns([2, 1, 1, 1])
                col_frame.image(annotated_frame, channels="RGB", use_container_width=True)
                col_m1.metric("Model 1 Score", f"{score1:.2f}")
                col_m1.markdown(f"**{status1}**")
                col_m2_graph.metric("Model 2 Score", f"{score2:.2f}")
                col_m2_graph.markdown(f"**{status2}**")
                col_m3.metric("Model 3 Score", f"{score3:.2f}")
                col_m3.markdown(f"**{status3}**")
                
                if len(st.session_state.engagement_history1) > 1:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    times = list(range(len(st.session_state.engagement_history1)))
                    ax.plot(times, list(st.session_state.engagement_history1), 'b-', label='Model 1', linewidth=2)
                    ax.plot(times, list(st.session_state.engagement_history2), 'orange', label='Model 2', linewidth=2)
                    ax.plot(times, list(st.session_state.engagement_history3), 'g-', label='Model 3', linewidth=2)
                    ax.set_ylim(0, 1)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    col_m3.pyplot(fig)
            else:
                features = extract_features(results.pose_landmarks.landmark, current_model)
                engagement_score = current_model.predict_proba(features.reshape(1, -1))[0][1]
                status = "Engaged" if engagement_score > 0.5 else "Not Engaged"
                
                st.session_state.engagement_history.append(engagement_score)
                st.session_state.engagement_history1.append(engagement_score)
                st.session_state.engagement_history2.append(engagement_score)
                st.session_state.engagement_history3.append(engagement_score)
                st.session_state.frame_count += 1
                
                score_placeholder.metric("Engagement Score", f"{engagement_score:.2f}", delta=None)
                status_placeholder.markdown(f"**Status:** {status}")
                
                # Graph
                if len(st.session_state.engagement_history) > 1:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    times = list(range(len(st.session_state.engagement_history)))
                    ax.plot(times, list(st.session_state.engagement_history), 'b-', linewidth=2)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Engagement Score")
                    ax.set_xlabel("Frames")
                    ax.set_title("Real-time Engagement Graph")
                    ax.grid(True, alpha=0.3)
                    graph_placeholder.pyplot(fig, use_container_width=True)
        else:
            engagement_score = 0.0
            status = "No Pose Detected"
            score_placeholder.metric("Engagement Score", "0.00", delta=None)
            status_placeholder.markdown("**No Pose Detected**")
        
        # Small delay for real-time feel
        time.sleep(0.03)
    
    cap.release()
else:
    st.info("☝️ Start live detection using the checkbox in the sidebar.")

# Footer
st.markdown("---")
st.markdown("**Research Prototype** | Built with Streamlit, MediaPipe, OpenCV, scikit-learn")

# ===== NEW VIDEO ANALYSIS SECTION =====
st.markdown("---")
st.header("🎥 Video Batch Analysis")
st.markdown("Upload videos for engagement analysis (processes directly, no preview)")

uploaded_files = st.file_uploader("Choose video files", type=['mp4','avi','mov','mkv'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        video_name = Path(uploaded_file.name).stem
        
        # Save uploaded bytes to temp video file
        temp_path = f"temp_{video_name}.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner(f"Processing {video_name}..."):
            if selected_model == "Compare Both":
                scores_dict = process_video_multi(temp_path, models)
                png1 = generate_engagement_graph(scores_dict['model1'], f"{video_name}_model1")
                png2 = generate_engagement_graph(scores_dict['model2'], f"{video_name}_model2")
                png3 = generate_engagement_graph(scores_dict['model3'], f"{video_name}_model3")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                frames = np.arange(len(scores_dict['model1']))
                window = 10
                smoothed1 = np.convolve(scores_dict['model1'], np.ones(window)/window, mode='valid')
                smoothed2 = np.convolve(scores_dict['model2'], np.ones(window)/window, mode='valid')
                smoothed3 = np.convolve(scores_dict['model3'], np.ones(window)/window, mode='valid')
                smoothed_frames = frames[:len(smoothed1)]
                ax.plot(frames, scores_dict['model1'], 'b-', alpha=0.5, label='Model1 Raw')
                ax.plot(frames, scores_dict['model2'], 'orange', alpha=0.5, label='Model2 Raw')
                ax.plot(frames, scores_dict['model3'], 'g-', alpha=0.5, label='Model3 Raw')
                ax.plot(smoothed_frames, smoothed1, 'b-', linewidth=2.5, label='Model1 Smoothed')
                ax.plot(smoothed_frames, smoothed2, 'orange', linewidth=2.5, label='Model2 Smoothed')
                ax.plot(smoothed_frames, smoothed3, 'g-', linewidth=2.5, label='Model3 Smoothed')
                ax.set_ylim(0, 1)
                ax.set_title(f"Model Comparison: {video_name}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                comp_png = f"comparison_{video_name}.png"
                plt.savefig(comp_png, dpi=300, bbox_inches='tight')
                plt.close()
                
                table_df = generate_evaluation_table(scores_dict, video_name, "Compare Both")
                st.subheader(f"Results for {video_name} (All Models)")
                st.dataframe(table_df)
                if png1: st.image(png1, caption="Model 1", use_container_width=True)
                if png2: st.image(png2, caption="Model 2", use_container_width=True)
                if png3: st.image(png3, caption="Model 3", use_container_width=True)
                st.image(comp_png, caption="Comparison", use_container_width=True)
            else:
                key_map = {'Model 1 (Original)': 'model1', 'Model 2 (New)': 'model2', 'Model 3 (Ensemble)': 'model3'}
                model_key = key_map.get(selected_model, 'model1')
                scores = process_video_multi(temp_path, models)[model_key]
                png_path = generate_engagement_graph(scores, video_name)
                table_df = generate_evaluation_table({model_key: scores}, video_name, selected_model)
                st.subheader(f"Results for {video_name} ({selected_model})")
                st.dataframe(table_df)
                if png_path:
                    st.image(png_path, caption=f"Engagement Graph: {video_name}", use_container_width=True)
        
        # Cleanup temp file
        os.remove(temp_path)
        st.success(f"✅ Completed {video_name}")

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. Upload MP4/AVI videos 
    2. Analysis runs automatically
    3. Graphs saved as PNG + table appended to `evaluation_results.csv`
    4. Direct processing - no video preview shown
    """)

