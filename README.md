# TALOS: Tactical Aerospace Localization & Objective Scoring

TALOS (Tactical Aerospace Localization & Objective Scoring) is an end-to-end tactical surveillance pipeline engineered for satellite and aerial imagery. It transforms raw pixels into actionable intelligence by fusing YOLOv11 object detection with a Kalman Filter and the Hungarian Algorithm for persistent multi-object tracking.

The system’s core is a predictive Threat Assessment Engine that ranks targets in real time. By calculating temporal urgency (Time-to-Impact), directional intent vectors, and spatial proximity, TALOS continuously updates a dynamic risk score for every detected aircraft. Designed for low-latency environments, it provides a robust framework for autonomous aerospace monitoring and predictive threat analysis.


##  Overview

Modern aerospace monitoring requires more than object detection. TALOS transforms raw surveillance feeds into actionable intelligence through:

- High-precision aircraft detection  
- Persistent identity tracking  
- Predictive threat scoring  
- Structured analytical logging  

---

## Project Demonstration

Below is the comparison between the raw surveillance feed and the processed tactical output.

| **Raw Input Feed** | **TALOS Tactical Output** |
| :---: | :---: |
| *Original Satellite Footage* | *Detection, Tracking & Threat Analysis Overlay* |

---

## System Architecture

### 1) Detection Engine — YOLOv11

- Custom-trained YOLOv11 model  
- Optimized for small-object detection in high-resolution satellite imagery   
---

### 2) Tracking Engine — Kalman Filter + Hungarian Algorithm

TALOS uses a dual-stage data association pipeline:

- **Kalman Filter**
  - Constant velocity motion model  
  - Predicts next position and velocity  

- **Hungarian Algorithm (Linear Sum Assignment)**
  - Minimizes global Euclidean distance  
  - Optimal detection-to-track matching  
  - Maintains persistent object IDs  

---

### 3) Threat Assessment Engine

The system calculates a dynamic risk score $S$ for every tracked object. This allows the pipeline to prioritize targets that pose the most immediate tactical threat.

#### Tactical Formulas

* **Time-to-Impact (TTI) — Urgency Estimation** Measures how soon a target will reach the protected origin based on its current closing velocity.
  $$TTI = \frac{d}{v_{close}}$$
  $$\text{Urgency} = e^{-\left(\frac{TTI}{\tau}\right)}$$
  *(Where $\tau$ is the sensitivity constant set to 5.0)*

* **Intent Vector Analysis — Directional Intent** Uses Cosine Similarity to determine if the aircraft's velocity vector $\vec{V}$ is aligned with the direction vector $\vec{D}$ toward the origin.
  $$\text{Intent} = \max\left(0, \frac{\vec{V} \cdot \vec{D}}{\|\vec{V}\| \|\vec{D}\|}\right)$$

* **Proximity Score — Distance-based Risk** An exponential escalation based purely on the Euclidean distance $d$ from the origin.
  $$\text{Proximity} = e^{-\left(\frac{d}{\sigma}\right)}$$

* **Certainty Score — Reliability Weight** Uses the **Trace of the Kalman Filter Covariance Matrix ($P$)** to reward stable, long-term tracks and filter out transient "flicker" detections.
  $$\text{Certainty} = e^{-\left(\frac{\text{Tr}(P)}{\lambda}\right)}$$

#### Final Scoring Logic
Outputs are frame-level and continuously updated using a weighted fusion of the metrics above:

$$S = (0.4 \times \text{Urgency}) + (0.3 \times \text{Proximity}) + (0.2 \times \text{Intent}) + (0.1 \times \text{Certainty})$$

---

## Project Structure

1. Update `config/config.yaml`
2. Update `src/TALOS/entity/`
3. Update `src/TALOS/config/`
4. Update `src/TALOS/components/`
5. Update `src/TALOS/pipeline/`
6. Update `main.py` 

After changes, run:

```bash
python3 main.py

 ---

##  Getting Started

### Prerequisites
- Python 3.12+
- NVIDIA GPU / Apple Silicon (MPS) / CPU

### Installation

```bash
git clone https://github.com/Gakshith/Tactical-Aerospace-Localization-Objective-Scoring-TALOS-
cd TALOS
pip install -r requirements.txt

## How to Run

### 1)Add Your Video
Place your video inside the `source_video/` folder and rename it to: input.mp4
### 2) Run the Pipeline : python3 main.py
### 3) Get the Results :After execution, outputs will be available in:
artifacts/model_run/ with output.mp4 file
output.mp4 — Processed tactical video
alos_threat_analysis.csv — Threat analysis log


