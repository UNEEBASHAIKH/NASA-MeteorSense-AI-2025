# Meteor Madness: AI-Powered Fireball Detection

A real-time meteor detection and analysis system built for the NASA Space Apps Challenge 2025. This pipeline processes sky camera feeds to identify meteor events, calculate trajectories, and classify celestial objects with machine learning precision.

---

## 🎯 Project Overview

**Challenge:** Meteor Madness  
**Team:** ASTROBLITZ  
**Timeline:** NASA Space Apps 2025 (48-hour hackathon)

Traditional meteor detection systems struggle with false positives and delayed analysis. Our solution leverages modern computer vision and real-time data processing to create an accessible, accurate detection platform for researchers and enthusiasts.

---

## 🚀 Core Features

- **Real-time Detection**: AI models process live camera feeds with <5 second latency
- **Trajectory Analysis**: Calculate meteor paths and potential impact zones
- **False Positive Filtering**: Advanced noise reduction eliminates 85%+ of common false alerts
- **Web Dashboard**: Interactive interface for monitoring and analysis
- **Modular Architecture**: Easy integration with existing observatory systems

---

##  Technical Stack

### Backend & AI
- **Computer Vision**: YOLO-v8 + OpenCV
- **API Framework**: FastAPI with WebSocket support
- **Data Processing**: NumPy, SciPy, Pandas
- **Task Queue**: Redis + Celery

### Dashboard & Visualization
- **Framework**: Streamlit
- **Charts**: Plotly, Altair
- **3D Visualization**: PyVista / Plotly 3D
- **Image Processing**: PIL, OpenCV

### Deployment
- **Containerization**: Docker + Docker Compose
- **GPU Acceleration**: CUDA-enabled containers

---

## 📁 Project Structure
