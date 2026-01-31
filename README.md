# AI-Based Space Debris Monitoring

## 🛰 Project Overview
This project implements a Physics-Informed Neural Network to assess and classify space debris risk based on orbital mechanics. The model combines orbital physics features with machine learning, enabling accurate classification of debris into risk categories:

- CRITICAL – Immediate reentry risk
- HIGH – High atmospheric drag, unstable orbit
- MEDIUM – Moderate risk
- LOW – Stable, high-altitude debris

The system can parse Two-Line Element (TLE) data, extract physics features, and provide risk predictions for individual objects.

The model was designed to:
- Analyze real orbital data (TLE)
- Apply orbital mechanics and atmospheric drag physics
- Integrate physics-informed machine learning
- Assess and classify space debris risk in near-Earth orbit

This work represents a prototype-level research and engineering solution developed under hackathon constraints.


## 📐 Architecture

### 1. TLE Parsing
- Reads standard TLE files (groups of 3 lines per object).
- Extracts key orbital elements:
  - Epoch, mean motion and derivatives
  - Inclination, RAAN, eccentricity, argument of perigee, mean anomaly
- Converts elements to physical quantities:
  - Semi-major axis
  - Perigee & apogee altitudes
  - Orbital period
  - Orbital velocities

### 2. Physics Feature Extraction
- Implements FinalPhysicsFeatures.calculate_physics_features.
- Key features derived from physics:

| Feature | Physical Meaning |
|---------|----------------|
| Perigee Altitude | Distance from Earth, determines drag and reentry risk |
| Drag Factor | Sensitivity to atmosphere; low orbit → higher drag |
| Eccentricity | Orbital stability; eccentric orbits increase risk of collision |
| Altitude Difference | Shape of orbit; more elongated orbits have higher velocity variation |
| BSTAR Factor | Ballistic coefficient, indicates drag susceptibility |
| Orbital Period | Time for one orbit; related to altitude and speed |
| Inclination Risk | Polar orbits are more congested → higher collision probability |
| Combined Risk | Weighted combination of factors for ML input |

### 3. Risk Classification
- Physics-informed thresholds:
  - CRITICAL: perigee < 200 km
  - HIGH: 200 km ≤ perigee < 350 km or high eccentricity / drag
  - MEDIUM: 350 km ≤ perigee < 700 km with moderate risk factors
  - LOW: high altitude, stable orbit
- Neural network also learns complex interactions between features.

### 4. Neural Network – FinalPhysicsNet
- Input: 8 physics-informed features
- Hidden layers: [128 → 128 → 64] with BatchNorm + ReLU + Dropout
- Output: 4 classes (CRITICAL, HIGH, MEDIUM, LOW)
- Loss: Weighted Cross-Entropy for class imbalance
- Optimizer: AdamW + OneCycleLR
- Supports GPU acceleration with PyTorch

### 5. Training Pipeline
- Parse TLE → extract physics features → encode labels → scale features
- Stratified train-test split
- Compute class weights to handle imbalanced debris data
- Early stopping on validation accuracy
- Outputs risk classification with confidence probabilities

### 6. Fallback Model
- If neural network fails or isn’t trained:
  - Uses perigee altitude heuristics
  - Provides approximate risk prediction
  - Ensures robust operation even without ML



## 🌌 Physics Principles Behind the Model

The program integrates orbital mechanics and atmospheric physics:

1. Keplerian Mechanics
   - Semi-major axis (a), eccentricity (e), and inclination (i) define orbital shape and speed.
   - Velocity at perigee/apogee calculated via: $v = \sqrt{\mu \left( \frac{2}{r} - \frac{1}{a} \right)}$

   where $mu$ is Earth’s gravitational constant, $r$ is distance from Earth, and $a$ is semi-major axis.

2. Atmospheric Drag
   - Low perigee → stronger drag → orbit decay
   - BSTAR factor models sensitivity to drag

3. Collision Risk
   - Polar orbits (inclination ~90°) are more crowded → higher chance of collisions
   - Eccentric orbits and large altitude differences increase relative velocities → higher risk

4. Reentry Prediction
   - Orbits with perigee < 200 km are considered CRITICAL due to rapid decay and reentry potential


## 🛠 Tools, Libraries, and Data Sources

### Programming Language
- Python 3.9+  
  Core language used for data processing, physics calculations, and model training.


### Machine Learning & Deep Learning
- PyTorch  
  Used to implement and train the neural network (FinalPhysicsNet), including GPU acceleration, loss functions, and optimization.  

- scikit-learn  
  Used for feature scaling, label encoding, dataset splitting, class weighting, and evaluation metrics.  

- NumPy  
  Used for numerical computations, vectorized physics calculations, and array manipulation.  


### Data Handling & Processing
- Pandas  
  Used to store, process, and analyze parsed TLE data in tabular form.  



### Orbital Data & Space Environment
- Two-Line Element (TLE) Data Format  
  Standard orbital element format used to describe satellite and debris orbits.  
  Provided by organizations such as NORAD and CelesTrak.

- CelesTrak  
  Source of up-to-date satellite and space debris TLE datasets.  


## 👥 Team
  Team name:  Ygddrasil  
  Event: ActInSpace Hackathon
