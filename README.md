# AI-Based Public Safety Monitoring & Risk Detection System

A full-stack intelligent system that analyzes crowd-level behavior from video input using **deep learning** and **computer vision** to produce a **timestamped risk timeline**. When risk escalates to **MEDIUM** or **HIGH**, the backend automatically creates a **police alert** (persisted on disk) that appears in the Police Dashboard with AI-generated emergency response recommendations.

This project focuses on **crowd motion / scene-level anomaly detection**. It does **not** perform face recognition or individual identification.

---

## 🎯 Background

In crowded public environments (markets, festivals, metro stations), incidents like panic, stampedes, sudden crowd movement, or accidents can escalate quickly if not detected early.

Traditional CCTV monitoring is:
- **Slow to respond** - Human operators can miss critical events
- **Prone to human error** - Fatigue and attention limitations
- **Difficult to scale** - Cannot monitor multiple camera feeds simultaneously
- **Reactive** - Responds after incidents occur, not before

This system addresses these challenges by converting video input into **automated risk classification** with **time-based evidence** and **AI-powered decision support**.

---

## 🚀 Project Objectives

Design and implement an intelligent AI system that:
- ✅ Accepts video input (pre-recorded or real-time streaming)
- ✅ Detects abnormal crowd behavior using multiple ML algorithms
- ✅ Classifies public safety risk as **NONE / LOW / MEDIUM / HIGH**
- ✅ Produces a **timeline** of risk windows with **time + cause + confidence**
- ✅ Triggers **police alerts** automatically for MEDIUM/HIGH risk
- ✅ Generates **emergency response recommendations** using AI
- ✅ Tracks user locations via GPS for rapid response
- ✅ Provides real-time visual feedback during analysis

---

## 📊 System Architecture Overview

### **Three-Tier ML Detection Pipeline**

```
VIDEO INPUT
    ↓
┌───────────────────────────────────────────┐
│   ANALYZER SELECTION (User Choice)        │
├───────────┬───────────────┬───────────────┤
│ Autoencoder│     YOLO      │ Optical Flow  │
│ (Deep CNN) │ (Detection)   │ (Motion)      │
└─────┬─────┴───────┬───────┴───────┬───────┘
      │             │               │
      └─────────────┴───────────────┘
                    ↓
        ┌───────────────────────┐
        │ RISK REASONING ENGINE │
        │ • Temporal Smoothing  │
        │ • State Machine       │
        │ • Confidence Scoring  │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │   ALERT GENERATION    │
        │ if risk ≥ MEDIUM      │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │  EMERGENCY RESPONSE   │
        │  (Gemini AI / Rules)  │
        └───────────────────────┘
```

---

## What You Get

### User Dashboard
- Upload video
- Configure sampling + thresholds
- View:
  - Overall risk and score
  - Event time (first MEDIUM/HIGH window)
  - Timeline entries: **Risk Level / Time / Cause**

### Police Dashboard
- Polls alerts every 3 seconds
- Displays user email + location + event time + risk score + cause
- Allows acknowledging alerts
- Alerts persist across restarts (stored in JSON)

---

## Architecture

- **Frontend**: Vite + React + Tailwind
- **Backend**: FastAPI (Python)
- **Analyzer (default in UI)**: TensorFlow/Keras autoencoder (pretrained)
- **Optional analyzer**: OpenCV optical-flow spike detector

Flow:
1. User uploads video in the frontend
2. Frontend sends a multipart request to the backend
3. Backend runs an analyzer and returns:
   - Overall risk
   - Timestamped samples
4. If any sample is MEDIUM/HIGH, backend creates an alert
5. Police dashboard polls and shows alerts

---

## Repository Structure

- [backend/app/main.py](backend/app/main.py): FastAPI routes (`/api/analyze`, `/api/alerts`, ack)
- [backend/app/analyzers/autoencoder.py](backend/app/analyzers/autoencoder.py): Autoencoder-based analysis
- [backend/app/analyzers/optical_flow.py](backend/app/analyzers/optical_flow.py): Optical-flow-based analysis
- [backend/app/storage.py](backend/app/storage.py): JSON persistence for alerts
- [backend/data/alerts.json](backend/data/alerts.json): Persisted police alerts
- [frontend](frontend): Vite/React UI (User + Police dashboards)
- [frontend/.env](frontend/.env): Demo credentials + API base URL + defaults
- [Crowd_Anomaly_Detection/AnomalyDetector.h5](Crowd_Anomaly_Detection/AnomalyDetector.h5): Pretrained model used by the autoencoder analyzer
- [Crowd_Anomaly_Detection/run_video_risk_alerts.py](Crowd_Anomaly_Detection/run_video_risk_alerts.py): Model utilities used by the backend analyzer

---

## Prerequisites

### Required
- Python 3.10+ (recommended for best TensorFlow compatibility)
- Node.js 18+ (or newer)

### Notes (Windows)
- TensorFlow installation can be heavy; ensure you have enough disk space.
- NumPy is pinned to `<2` because many TensorFlow builds still require it.

---

## Setup (Backend)

From the repo root:

1) Create/activate a virtual environment (optional but recommended)

PowerShell:
- Create: `python -m venv .venv`
- Activate: `.\.venv\Scripts\Activate.ps1`

2) Install dependencies

- Backend API deps: `pip install -r backend/requirements.txt`
- Model/analyzer deps: `pip install -r Crowd_Anomaly_Detection/requirements.txt`

Notes:
- The API uses multipart form upload; ensure `python-multipart` is installed (it is included in `backend/requirements.txt`).
- If you use a newer Python version and TensorFlow fails to install/import, use Python 3.10/3.11 for best compatibility.

3) Run the API

- `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`

Optional (more console logs):
- `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000 --log-level debug`

Health check:
- Open `http://127.0.0.1:8000/api/health`

Uploads are stored under:
- [backend/uploads](backend/uploads)

Alerts persist to:
- [backend/data/alerts.json](backend/data/alerts.json)

---

## Setup (Frontend)

1) Go to the frontend folder
- `cd frontend`

2) Install dependencies
- `npm install`

3) Configure environment variables

Edit demo values in:
- [frontend/.env](frontend/.env)

4) Run the dev server
- `npm run dev`

Open:
- `http://localhost:5173`

---

## Environment Variables (Frontend)

Configured in [frontend/.env](frontend/.env):

- `VITE_API_BASE_URL` (default: `http://127.0.0.1:8000`)
- `VITE_DEFAULT_LOCATION` (default: `kandivali`)
- `VITE_FORCE_LOCATION` (`true` or `false`)

---

## 🧠 Machine Learning & AI Architecture

### **1. Autoencoder-Based Anomaly Detection**

#### **Architecture: Convolutional LSTM Autoencoder**

**Model Type:** Unsupervised Deep Learning - Spatial-Temporal Reconstruction

**Network Structure:**
```
Input: (227×227×10×1) - 10 grayscale frames

ENCODER (Compression):
├─ Conv3D(128 filters, 11×11×1 kernel, stride 4×4×1, tanh)
├─ Conv3D(64 filters, 5×5×1 kernel, stride 2×2×1, tanh)
├─ ConvLSTM2D(64 filters, dropout=0.4, recurrent_dropout=0.3)
├─ ConvLSTM2D(32 filters, dropout=0.3) ← Bottleneck
└─ ConvLSTM2D(64 filters, dropout=0.5)

DECODER (Reconstruction):
├─ Conv3DTranspose(128 filters, 5×5×1 kernel, stride 2×2×1, tanh)
└─ Conv3DTranspose(1 filter, 11×11×1 kernel, stride 4×4×1, tanh)

Output: (227×227×10×1) - Reconstructed frames
```

**Why This Architecture?**

1. **Conv3D Layers**: Process spatial (x,y) and temporal (t) dimensions simultaneously
   - Captures motion patterns across consecutive frames
   - Parameter efficiency: ~99% fewer parameters than regular LSTM

2. **ConvLSTM2D**: Maintains temporal context through recurrent connections
   - Learns motion trajectories: velocity, acceleration patterns
   - Preserves spatial structure (regular LSTM loses pixel neighborhoods)

3. **Training Paradigm**: 
   - Trained on **normal crowd behavior only**
   - Model learns to reconstruct "normal" patterns with low error
   - Abnormal behavior → reconstruction fails → **high loss = anomaly**

**Loss Function: Mean Euclidean Distance**

```python
Loss = √(Σ(input - output)²) / N
where N = 227 × 227 × 10 = 515,290 pixels
```

**Why Mean Euclidean vs MSE?**
- **Interpretability**: Linear scale vs quadratic
- **Threshold Stability**: Easier to set thresholds (10⁻³ vs 10⁻⁶ scale)
- **Gradient Behavior**: More stable training

**Risk Thresholds:**
- Loss < 0.0008 → NONE/LOW (normal crowd)
- Loss 0.0008-0.0012 → MEDIUM (elevated risk)
- Loss > 0.0016 → HIGH (critical risk)

**Preprocessing Pipeline:**
```python
1. Extract frames every 0.2s (configurable)
2. Convert to grayscale: gray = 0.2989×R + 0.5870×G + 0.1140×B
3. Resize to 227×227
4. Z-score normalization: (x - μ) / σ
5. Bundle into 10-frame sequences
```

---

### **2. YOLOv8 Object Detection + Crowd Analytics**

#### **Detection Algorithm: Single-Shot Multi-Scale Detection**

**Model:** YOLOv8n (Nano version, 6.2 MB)
- **Speed**: 120 FPS on GPU, real-time capable
- **Accuracy**: 37.3 mAP (sufficient for person counting)
- **Why YOLO vs Faster R-CNN?** 4× faster, single-shot detection

**Detection Process:**
```python
1. YOLO person detection (class=0, confidence>0.4)
2. Extract bounding boxes: [x1, y1, x2, y2, confidence]
3. Calculate centroids: [(x1+x2)/2, (y1+y2)/2]
4. Multi-frame tracking using IoU (Intersection over Union)
```

**IoU Tracking Algorithm:**
```python
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)

If IoU > 0.3 between frames:
    → Same person (update track)
Else:
    → New person (create track)
```

**Why IoU > 0.3?**
- 30% overlap indicates same object across frames
- Too low (< 0.2): Spurious matches, track fragmentation
- Too high (> 0.5): Lost tracks during fast motion

#### **Crowd Metrics Calculation**

**1. Density Score**
```python
Density = Σ(bbox_areas) / frame_area
Range: [0, 1]
0.3 = moderate, 0.7+ = severe overcrowding
```

**2. Motion Speed**
```python
Speed = Σ distance(current, previous) × FPS / N
Units: pixels/second
Normalized to [0, 1] using max_speed = 200 px/s
```

**3. Direction Entropy (Shannon Entropy)**
```python
H = -Σ p(θ) × log₂(p(θ))
where θ = movement angles (16 bins, 22.5° each)

Normalized: H_norm = H / log₂(16) = H / 4.0
```

**Why Logarithm?**
- Information theory: Entropy measured in **bits**
- log₂(16) = 4 bits needed to encode 16 equally-likely outcomes
- Detects uniform chaos (panic) vs organized movement (evacuation)

**Physical Interpretation:**
- H_norm = 0.0: Organized evacuation (all moving same direction)
- H_norm = 0.5: Moderate randomness (normal mixing)
- H_norm = 0.9: Panic scenario (erratic movement in all directions)

**4. Spatial Spread**
```python
σ_x = √(1/n × Σ(x_i - μ_x)²)  # Standard deviation in x
σ_y = √(1/n × Σ(y_i - μ_y)²)  # Standard deviation in y

Spread = (σ_x + σ_y) / frame_diagonal

Normalized by diagonal for resolution-independence
```

---

### **3. Optical Flow Analysis (Farneback Algorithm)**

#### **Dense Motion Vector Calculation**

**Algorithm:** Farneback Dense Optical Flow
- **Type**: Computer vision motion estimation
- **Output**: H×W motion vectors (every pixel)
- **Why Dense vs Sparse?** Captures crowd motion everywhere, not just feature points

**Mathematical Foundation:**

```
Optical Flow Constraint Equation:
I_x · u + I_y · v + I_t = 0

where:
u, v = horizontal, vertical velocities
I_x, I_y, I_t = spatial and temporal gradients
```

**Farneback Solution: Polynomial Expansion**
```python
Approximate neighborhood with quadratic:
f(x) ≈ x^T A x + b^T x + c

Displacement d minimizes:
||f₂(x) - f₁(x - d)||²

Closed-form solution:
d = -(A₁ + A₂)⁻¹ · (b₂ - b₁)/2
```

**Algorithm Parameters:**
```python
cv2.calcOpticalFlowFarneback(
    prev, next,
    pyr_scale=0.5,     # Pyramid downsampling factor
    levels=3,          # 3 pyramid levels
    winsize=15,        # 15×15 averaging window
    iterations=3,      # 3 iterations per level
    poly_n=5,          # 5×5 polynomial neighborhood
    poly_sigma=1.2     # Gaussian smoothing σ=1.2
)
```

**Why These Parameters?**
- **pyr_scale=0.5**: Creates pyramid [1×, 0.5×, 0.25×, 0.125×]
- **levels=3**: Detects motion from 1 to ~120 pixels
- **winsize=15**: Balance between smoothness and detail
- **poly_n=5**: Captures local motion structure

#### **Statistical Anomaly Detection**

**Robust Statistics: MAD (Median Absolute Deviation)**

```python
MAD = median(|x_i - median(x)|)
MAD_scaled = MAD × 1.4826

Why MAD instead of Standard Deviation?
- Breakdown point: 50% (vs 0% for σ)
- Robust to outliers (single runner doesn't affect crowd baseline)
```

**Z-Score Calculation:**
```python
Z = (current_flow - median) / (MAD × 1.4826)

Thresholds:
Z < 3.0:  Normal (99.7% of data under normal distribution)
Z = 3.0:  LOW risk (3σ outlier, p=0.003)
Z = 5.0:  MEDIUM risk (5σ outlier, p=5.7×10⁻⁷)
Z = 7.0:  HIGH risk (7σ outlier, p=1.3×10⁻¹²)
```

**Why These Thresholds?**
- Based on standard normal distribution properties
- Baseline walking: z ≈ 1-2
- Jogging crowd: z ≈ 3-4
- Stampede: z > 7

---

### **4. Risk Reasoning Engine**

#### **Temporal State Machine**

**Purpose**: Prevent false positives through multi-frame confirmation

**State Transitions:**
```
NONE → LOW:     1 consecutive LOW frame
NONE → MEDIUM:  2 consecutive MEDIUM frames
* → HIGH:       3 consecutive HIGH frames

Cooldown: Hold for 5 frames after escalation
De-escalation: Require 3 frames below threshold
```

**Mathematical Guarantee:**
```
False Positive Rate Reduction:
P(false_HIGH | noise) = P(spike)^confirm_frames

If P(single_spike) = 0.05 (5%)
Then P(false_HIGH) = 0.05³ = 0.0125% (400× reduction)
```

#### **Weighted Risk Scoring**

**Formula:**
```python
Risk_Score = 100 × Σ(weight_i × normalized_signal_i)

Signal normalization: s_norm = clamp((s - min)/(max - min), 0, 1)
```

**Weight Configurations:**

| Signal | Autoencoder | YOLO | Optical Flow |
|--------|-------------|------|--------------|
| motion_speed | 0.35 | 0.30 | 0.40 |
| directional_chaos | 0.05 | 0.25 | 0.00 |
| density_change_rate | 0.25 | 0.25 | 0.30 |
| spread | 0.10 | 0.20 | 0.30 |
| persistence | 0.25 | 0.00 | 0.00 |

**Why Different Weights?**
- **Autoencoder**: Can't track directions → low chaos weight
- **YOLO**: Explicit tracking → high chaos weight
- **Optical Flow**: Statistical motion → high speed weight

**Risk Level Mapping:**
```python
score ≤ 0:   NONE
score ≤ 30:  LOW
score ≤ 70:  MEDIUM
score > 70:  HIGH
```

#### **Confidence Calculation**

```python
confidence = agreement_factor × consistency_factor × persistence_factor

agreement_factor = signals_agreeing / total_signals
  (signal "agrees" if normalized value ≥ 0.55)

consistency_factor = exp(-std_dev / 20.0)
  (exponential decay based on variance)

persistence_factor = min(consecutive_frames / 10, 1.0)
  (saturates at 10 frames)
```

---

### **5. Emergency Response AI**

#### **Hybrid System: LLM + Rule-Based Fallback**

**Primary: Gemini 2.5 Flash Lite**

**Prompt Structure:**
```python
"""
Analyze this {risk_level}-risk incident:

INCIDENT DATA:
- Time: {HH:MM:SS}
- Location: {location}
- Risk Level: {MEDIUM/HIGH}
- Risk Score: {0-100}
- Confidence: {0-100%}
- Primary Cause: {cause}
- Supporting Factors: {[factors]}
- Duration: {seconds}s
- Escalation Trend: {increasing/stable/decreasing}

Output format:
Unit: [Response unit type]
Action: [Specific action]
Urgency: [Immediate/High/Medium]
Reasoning:
- [Point 1]
- [Point 2]
"""
```

**Fallback: Rule-Based Decision Tree**

```python
if "density" or "overcrowding" in primary_cause:
    unit = "Crowd Control Police"
    action = "Restrict entry, manage flow"
    urgency = "Immediate" if HIGH else "Medium"

elif "panic" or "stampede" in primary_cause:
    unit = "Tactical Response Team"
    action = "Create evacuation corridors"
    urgency = "Immediate"

elif "fire" or "smoke" in primary_cause:
    unit = "Fire Safety Team"
    action = "Immediate evacuation"
    urgency = "Immediate"

elif "medical" in primary_cause:
    unit = "Emergency Medical Services"
    action = "Clear medical access path"
    urgency = "Immediate" or "High"

# Duration escalation
if duration > 30s:
    urgency = upgrade(urgency)

if escalation_trend == "increasing":
    urgency = "Immediate"
```

**Why Hybrid Approach?**
1. **LLM**: Context-aware, adaptive reasoning
2. **Rules**: 100% predictable, <1ms latency
3. **Safety**: Fallback ensures response even if API fails

---

## 🔄 Complete System Workflow

### **Phase 1: Video Upload & Preprocessing**

```
1. User uploads video via React frontend
2. Form collects:
   - Video file (.mp4, .avi, .mov, .mkv)
   - User email, location, phone number
   - Analyzer choice (autoencoder/yolo/optical_flow)
   - Threshold parameters
3. Backend receives multipart form data
4. Video saved to backend/uploads/
```

### **Phase 2: ML Analysis Pipeline**

**Autoencoder Path:**
```
Video → Frame extraction (every 0.2s)
      → Grayscale conversion
      → Resize to 227×227
      → Z-score normalization
      → Group into 10-frame sequences
      → Autoencoder inference
      → Calculate reconstruction loss
      → Risk Reasoning Engine
      → Generate risk timeline
```

**YOLO Path:**
```
Video → Frame extraction (5 FPS)
      → YOLOv8 person detection
      → Multi-object tracking (IoU)
      → Calculate crowd metrics:
          • Density score
          • Motion speed
          • Direction entropy
          • Spatial spread
      → Risk Reasoning Engine
      → Generate risk timeline
```

**Optical Flow Path:**
```
Video → Frame extraction
      → Grayscale + resize (320px)
      → Farneback optical flow
      → Motion magnitude calculation
      → Statistical anomaly detection (Z-score)
      → Risk Reasoning Engine
      → Generate risk timeline
```

### **Phase 3: Risk Decision Making**

```
For each analyzed frame:
  1. Receive frame-by-frame signals
  2. Calculate weighted risk score
  3. Apply temporal smoothing (10-frame window)
  4. Confirm risk level (requires agreement)
  5. Track persistence and trends
  6. Generate RiskDecision:
     - Risk Score (0-100)
     - Risk Level (NONE/LOW/MEDIUM/HIGH)
     - Confidence (0-1)
     - Primary Cause
     - Supporting Factors
     - Explanation
```

### **Phase 4: Alert Generation**

```
If ANY frame reaches MEDIUM or HIGH:
  1. Create Alert object:
     - ID (UUID)
     - Timestamp
     - Event time (first MEDIUM/HIGH frame)
     - Overall risk level (highest detected)
     - Risk score, confidence
     - Primary cause + supporting factors
     - User email, location, video filename
  
  2. Persist to backend/data/alerts.json
  
  3. Generate emergency response (optional):
     - Call Gemini API with incident context
     - OR use rule-based fallback
     - Generate: unit, action, urgency, reasoning
  
  4. Return to frontend with full timeline
```

### **Phase 5: Police Dashboard Monitoring**

```
1. Police dashboard polls /api/alerts every 3 seconds
2. Backend returns unacknowledged alerts
3. Frontend displays:
   - Map view with GPS markers
   - Alert cards with:
     • Risk level badge (color-coded)
     • User email, location, phone
     • Timestamp, event time, risk score
     • Primary cause, supporting factors
     • Emergency response recommendation
   - Action buttons (Acknowledge, Share)

4. When acknowledged:
   - POST /api/alerts/{id}/acknowledge
   - Update acknowledged_at timestamp
   - Remove from active alerts list
```

---

## 📊 Key Parameters & Configuration

### **Detection Sensitivity**

```python
# Autoencoder (configurable in UI)
threshold_low = 0.0008     # Lower = more sensitive
threshold_medium = 0.0012
threshold_high = 0.0016    # Higher = less sensitive

# YOLO
confidence_threshold = 0.4  # Detection confidence (40%)
iou_threshold = 0.3        # Tracking threshold (30%)

# Optical Flow
z_low = 3.0     # 3σ outlier
z_med = 5.0     # 5σ outlier
z_high = 7.0    # 7σ outlier
```

### **Processing Performance**

```python
# Frame sampling
sample_every_seconds = 0.2  # Default: 5 samples/sec
process_fps = 10.0          # YOLO/Flow: 10 frames/sec

# Resize (speed optimization)
autoencoder_size = (227, 227)
optical_flow_size = (320, 240)
```

### **Risk Reasoning**

```python
# Temporal confirmation
window = 10                 # Sliding window size
require_agreement = 3       # Min signals agreeing
high_confirm_frames = 3     # Consecutive HIGH frames
medium_confirm_frames = 2   # Consecutive MEDIUM frames
cooldown_frames = 5         # Anti-oscillation delay
```

### **Alert Polling**

```javascript
// Police dashboard refresh
polling_interval = 3000  // milliseconds (3 seconds)
```

---

## 🎓 Technical Concepts Explained

### **1. Why ConvLSTM2D vs Regular LSTM?**

**Regular LSTM**: Input is flattened vector → loses spatial relationships
```
Input: 26×26 = 676-dimensional vector
Weights: 676² = 456,976 parameters per gate
```

**ConvLSTM**: Input is 2D grid → maintains pixel neighborhoods
```
Input: 26×26 grid
Kernel: 3×3 = 9 weights (reused via sliding)
Parameters: ~99% fewer
```

**Benefit**: Tracks motion spatially (person at [x,y] moving to [x+5,y+2])

### **2. Why Shannon Entropy for Chaos?**

**Not just counting directions:**
```
Scenario A: 90% North, 10% East
  Naive count: 2 directions
  Entropy: H = 0.47 bits → 0.12 normalized (LOW chaos)

Scenario B: Uniform distribution across 8 directions
  Naive count: 8 directions
  Entropy: H = 3.0 bits → 0.75 normalized (HIGH chaos)
```

**Entropy detects distribution shape, not just variety**

### **3. Why MAD vs Standard Deviation?**

**Outlier sensitivity:**
```
Normal: [2.1, 2.3, 2.0, 2.4, 2.2]
With outlier: [2.1, 2.3, 2.0, 2.4, 45.0]

Standard Deviation:
  σ_normal = 0.13
  σ_outlier = 16.1  (123× increase!)

MAD:
  MAD_normal = 0.15
  MAD_outlier = 0.20  (1.33× increase)
```

**MAD resists single outliers** (e.g., one person running while crowd walks)

### **4. Why State Machine vs Direct Thresholding?**

**Without state machine:**
```
Frame 100: score=25 → NONE
Frame 101: score=75 → HIGH (ALERT!)  ← False positive
Frame 102: score=20 → NONE

Result: 1 false alert
```

**With state machine (3 consecutive):**
```
Frame 100: score=25 → consec_high=0 → NONE
Frame 101: score=75 → consec_high=1 → NONE (pending)
Frame 102: score=20 → consec_high=0 → NONE

Result: No alert (correctly filtered)
```

**400× reduction in false positives**

---

## Repository Structure

- [backend/app/main.py](backend/app/main.py): FastAPI routes (`/api/analyze`, `/api/alerts`, `/api/emergency-response`)
- [backend/app/analyzers/](backend/app/analyzers/): ML analyzers (autoencoder, YOLO, optical flow)
- [backend/app/risk_reasoning.py](backend/app/risk_reasoning.py): Risk scoring engine with state machine
- [backend/app/emergency_response.py](backend/app/emergency_response.py): AI-powered emergency response generator
- [backend/app/storage.py](backend/app/storage.py): JSON persistence for alerts
- [backend/data/alerts.json](backend/data/alerts.json): Persisted police alerts
- [Crowd_Anomaly_Detection/AnomalyDetector.h5](Crowd_Anomaly_Detection/AnomalyDetector.h5): Pretrained ConvLSTM autoencoder model
- [frontend/src/pages/](frontend/src/pages/): User Dashboard, Police Dashboard, Login
- [frontend/src/components/Map.jsx](frontend/src/components/Map.jsx): Google Maps integration with GPS tracking

---

## Prerequisites

### Required
- Python 3.10+ (recommended for best TensorFlow compatibility)
- Node.js 18+ (or newer)
- Google Maps API Key (for location features)
- Gemini API Key (optional, for AI-powered emergency responses)

### Notes (Windows)
- TensorFlow installation can be heavy; ensure you have enough disk space
- NumPy is pinned to `<2` because many TensorFlow builds still require it

---

## Setup (Backend)

From the repo root:

1) Create/activate a virtual environment (optional but recommended)

PowerShell:
- Create: `python -m venv .venv`
- Activate: `.\.venv\Scripts\Activate.ps1`

2) Install dependencies

- Backend API deps: `pip install -r backend/requirements.txt`
- Model/analyzer deps: `pip install -r Crowd_Anomaly_Detection/requirements.txt`

3) Configure environment variables (optional)

Create `backend/.env`:
```
GEMINI_API_KEY=your_api_key_here
```

4) Run the API

- `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`

Health check:
- Open `http://127.0.0.1:8000/api/health`

Uploads are stored under:
- [backend/uploads](backend/uploads)

Alerts persist to:
- [backend/data/alerts.json](backend/data/alerts.json)

---

## Setup (Frontend)

1) Go to the frontend folder
- `cd frontend`

2) Install dependencies
- `npm install`

3) Configure environment variables

Create `frontend/.env`:
```
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_GOOGLE_MAPS_API_KEY=your_maps_api_key
VITE_DEFAULT_LOCATION=Kandivali
```

4) Run the dev server
- `npm run dev`

Open:
- `http://localhost:5173`

---

## API Reference

### `GET /api/health`
Returns server status.

### `POST /api/analyze`
Multipart form upload for video analysis.

Form fields (common):
- `file` (video: `.mp4`, `.avi`, `.mov`, `.mkv`)
- `userEmail` (string)
- `location` (string)
- `analyzer` (`autoencoder` | `yolo` | `optical_flow`)

Autoencoder parameters:
- `sampleEverySeconds` (default `0.2`)
- `thresholdLow` (default `0.0008`)
- `thresholdMedium` (default `0.0012`)
- `thresholdHigh` (default `0.0016`)

Response:
- `riskLevel`: `NONE | LOW | MEDIUM | HIGH`
- `riskScore`: numeric score (0-100)
- `confidence`: confidence metric (0-1)
- `eventTimeSeconds`: first MEDIUM/HIGH detection
- `samples`: timeline array with risk data per frame
- `alertCreated`: boolean
- `alert`: alert object (if created)

### `GET /api/alerts`
Returns persisted alerts (unacknowledged by default).

### `POST /api/alerts/{id}/acknowledge`
Marks an alert as acknowledged.

### `POST /api/alerts/{id}/emergency-response`
Generates emergency response recommendation using AI.

Parameters:
- `area_affected` (optional)
- `duration_seconds` (optional)
- `escalation_trend` (optional)

---

## Risk + Alerts Rules

- The analyzer returns a timeline of sampled windows
- If **any** sample window is **MEDIUM** or **HIGH**, the backend creates a police alert
- The alert includes:
  - User email, location, phone number
  - Risk level, score, confidence
  - Event time, primary cause, supporting factors
  - Emergency response recommendation (if generated)

---

## Troubleshooting

### Backend not starting: `python-multipart` missing
```bash
pip install -r backend/requirements.txt
```

### Port 8000 already in use
Option A — Kill process:
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Option B — Use different port:
```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001
```
Update `VITE_API_BASE_URL` in frontend/.env accordingly.

---

## 📚 Research References

- **ConvLSTM**: Shi et al. (2015) "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
- **Crowd Dynamics**: Helbing et al. (2002) "Simulation of pedestrian crowds in normal and evacuation situations"
- **Shannon Entropy**: Shannon (1948) "A Mathematical Theory of Communication"
- **Robust Statistics**: Rousseeuw & Croux (1993) "Alternatives to the Median Absolute Deviation"
- **YOLO**: Redmon et al. (2016) "You Only Look Once: Unified, Real-Time Object Detection"
- **Optical Flow**: Farnebäck (2003) "Two-Frame Motion Estimation Based on Polynomial Expansion"

---

## 🎯 Key Innovations

1. **Multi-Modal Detection**: Three complementary ML approaches (autoencoder, YOLO, optical flow)
2. **Temporal Reasoning**: State machine prevents false positives (400× reduction)
3. **Robust Statistics**: MAD-based anomaly detection resists outliers
4. **AI-Powered Response**: Gemini LLM generates contextual emergency recommendations
5. **Real-Time GPS**: Live location tracking for rapid emergency response
6. **Explainable AI**: Every decision includes primary cause, supporting factors, and confidence

---

## License

MIT

---

## Authors

AI-Based Public Safety Monitoring Team

---

## 🚀 Future Enhancements

- [ ] Multi-camera fusion for venue-wide monitoring
- [ ] Edge deployment on NVIDIA Jetson for on-device processing
- [ ] Historical incident database for pattern recognition
- [ ] Integration with emergency services dispatch systems
- [ ] Mobile app for first responders
- [ ] Thermal imaging support for low-light conditions
- `VITE_USER_EMAIL`
- `VITE_USER_PASSWORD`
- `VITE_POLICE_EMAIL`
- `VITE_POLICE_ID`
- `VITE_POLICE_PASSWORD`

---

## API Reference

### `GET /api/health`
Returns server status.

### `POST /api/analyze`
Multipart form upload.

Form fields (common):
- `file` (video: `.mp4`, `.avi`, `.mov`, `.mkv`)
- `userEmail` (string)
- `location` (string; default `kandivali`)
- `analyzer` (`autoencoder` or `optical_flow`)

Autoencoder parameters:
- `sampleEverySeconds` (default `0.2`)
- `thresholdLow` (default `0.0008`)
- `thresholdMedium` (default `0.0012`)
- `thresholdHigh` (default `0.0016`)
- `includeLosses` (`true/false`)

Optical-flow parameters:
- `processFps` (default `5.0`)
- `minConsecutive` (default `1`)
- `zLow` (default `3.0`)
- `zMed` (default `5.0`)
- `zHigh` (default `7.0`)

Response highlights:
- `riskLevel`: `NONE | LOW | MEDIUM | HIGH`
- `riskScore`: numeric score
- `eventTimeSeconds`: first time a MEDIUM/HIGH window occurred
- `samples`: array of timeline entries with `riskLevel`, `timeSeconds`, `cause` (and optional diagnostics)
- `alertCreated`: `true` if risk was MEDIUM/HIGH
- `alert`: created alert object (when `alertCreated=true`)

Example (curl):
- `curl -X POST "http://127.0.0.1:8000/api/analyze" -F "file=@your_video.mp4" -F "userEmail=user@example.com" -F "location=kandivali" -F "analyzer=autoencoder" -F "sampleEverySeconds=0.2"`

### `GET /api/alerts?includeAcknowledged=true|false`
Returns persisted alerts.

### `POST /api/alerts/{id}/ack`
Marks an alert as acknowledged.

---

## Risk + Alerts Rules

- The analyzer returns a timeline of sampled windows.
- If **any** sample window is **MEDIUM** or **HIGH**, the backend creates a police alert.
- The alert includes:
  - user email
  - location
  - risk level
  - risk score
  - event time seconds
  - cause

---

## Troubleshooting

### Backend not starting: `python-multipart` missing
If you see:
`Form data requires "python-multipart" to be installed.`

Install backend deps:
- `pip install -r backend/requirements.txt`

Or install only the missing package:
- `pip install python-multipart`

Then start again:
- `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`

### Backend not starting: `WinError 10048` (port 8000 already in use)
This means another process is already listening on `127.0.0.1:8000`.

Option A — find + kill the process (PowerShell):
- `netstat -ano | findstr :8000`
- Copy the PID from the last column, then:
- `taskkill /PID <PID> /F`

Option B — run backend on a different port:
- `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001`

If you change the port, update the frontend to match:
- Set `VITE_API_BASE_URL=http://127.0.0.1:8001` in [frontend/.env](frontend/.env)
- Restart the frontend dev server: `npm run dev`

### Frontend loads but API calls fail
- Verify backend is running at `http://127.0.0.1:8000`
- Verify `VITE_API_BASE_URL` in [frontend/.env](frontend/.env)
- If you change `.env`, restart `npm run dev`

### CORS error in browser console
- Default backend CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`.
- If your frontend runs on a different origin, update CORS in [backend/app/main.py](backend/app/main.py).

### TensorFlow import errors
- Ensure you installed [Crowd_Anomaly_Detection/requirements.txt](Crowd_Anomaly_Detection/requirements.txt)
- Ensure NumPy remains `<2`

### Alerts not persisting
- Confirm [backend/data](backend/data) exists and is writable
- Alerts are stored in [backend/data/alerts.json](backend/data/alerts.json)

---

## Security & Privacy Notes

- This is a demo system with simple credential checks in the frontend.
- Do not deploy as-is on the public internet.
- The intended scope is **crowd-level risk detection**, not personal surveillance.

---


VITE_POLICE_EMAIL=princemaurya8879@gmail.com
VITE_POLICE_ID=MH1234
VITE_POLICE_PASSWORD=1234
VITE_USER_EMAIL=prince@gmail.com
VITE_USER_PASSWORD=1234
VITE_DEFAULT_LOCATION=Kandavli
VITE_FORCE_LOCATION=true
VITE_GOOGLE_MAPS_API_KEY="AIzaSyBjhBQ6-WzlUqc2tIhpe1k10DpbgPWX_Lc"
VITE_API_BASE_URL=http://127.0.0.1:8000


