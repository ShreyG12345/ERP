# MSc Data Science Extended Research Project Plan

## Project Overview

**Student:** Shrey Goswami  
**Student Number:** 10903938  
**Project Title:** Machine Learning Analysis of Heterogeneous Intracellular Movement  
**Supervisor:** Prof. Sergei Fedotov and Dr. Nickolay Korabel  

---

## Executive Summary

This project investigates the potential of machine learning to enhance our understanding of intracellular substance flow within nerve cells. Proteins and organelles travel between cells to perform vital functions, and conditions such as Parkinson's and Alzheimer's can alter these movement patterns from their regular trajectories.

The research aims to identify cellular movement patterns and discover how these components behave differently across various diseases using experimental data from living cells. The approach involves preprocessing and cleaning data using machine learning techniques, followed by neural network analysis to detect abnormalities. The results could significantly improve our understanding of neurological illness causes and treatments.

---

## Research Objectives

### Primary Aim
Design and train an LSTM neural network capable of accurately predicting the Hurst Exponent and generalized diffusion coefficient from particle trajectories to characterize diffusive behavior.

### Specific Objectives

1. **Synthetic Data Generation & Model Training**
   - Generate synthetic fBm trajectories with varying Hurst exponents and generalized diffusion coefficients
   - Create controlled training datasets to validate model performance

2. **Experimental Data Application**
   - Apply the best-performing model to analyze experimental trajectories
   - Infer underlying diffusive regimes of endosomal motion

3. **Performance Evaluation**
   - Investigate how incorporating variable generalized diffusion coefficients influences model performance
   - Assess predictive accuracy and generalization capability

---

## Methodology

### Core Approach
This project constructs a deep learning model to predict the Hurst exponent and generalized diffusion coefficient from experimental single particle trajectories, following data collection methods from Han et al. (2020). The focus is on developing and training a Long Short-Term Memory (LSTM) neural network that can capture temporal dependencies and the diffusive nature of trajectory data.

### Technical Implementation
- **Platform:** Python-based modeling and analysis
- **Architecture:** LSTM neural networks with optimized hyperparameters
- **Loss Function:** Mean Squared Error (MSE) for continuous value prediction
- **Validation Metrics:** Mean Absolute Error (MAE) and R² for regression quality assessment

### Training Strategy
1. **Synthetic Dataset Generation**
   - fBm trajectories with H ∈ (0.1, 0.9)
   - Generalized diffusion coefficients D ∈ (0.1, 100)
   - Controlled training environment for model learning

2. **Model Validation**
   - Systematic variation of trajectory length and noise levels
   - Robustness testing across multiple validation sets
   - Selection of optimal model based on prediction error and consistency

3. **Experimental Analysis**
   - Application of best-performing LSTM model to original trajectories
   - Investigation of fundamental research questions regarding diffusive behavior

---

## Project Timeline

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| **Literature Review & Problem Definition** | Weeks 1-4 | Background research, methodology planning |
| **Synthetic Data Generation & Preprocessing** | Weeks 5-8 | fBm simulation, data preparation |
| **Model Design & Implementation** | Weeks 9-13 | LSTM architecture development |
| **Model Training & Validation** | Weeks 14-18 | Hyperparameter optimization, performance testing |
| **Experimental Data Analysis** | Weeks 19-22 | Real data application, pattern identification |
| **Result Interpretation & Discussion** | Weeks 23-25 | Biological insights, model evaluation |
| **Thesis Writing** | Weeks 26-29 | Documentation, analysis synthesis |
| **Final Review & Submission** | Weeks 30-32 | Quality assurance, final submission |

---

## Risk Assessment & Mitigation

### Identified Risks

1. **Data Quality Issues**
   - **Risk:** Experimental cargo trajectory data may contain noise, missing values, or inconsistencies
   - **Mitigation:** Implement robust data cleaning and preprocessing techniques early in the project

2. **Model Complexity Challenges**
   - **Risk:** Intracellular movement complexity may require extensive parameter fine-tuning
   - **Mitigation:** Test both feed-forward and recurrent neural networks for optimal model selection

3. **Interpretability Limitations**
   - **Risk:** Deep learning models may lack interpretability for meaningful biological insights
   - **Mitigation:** Incorporate AI techniques and visualization tools for enhanced understanding

4. **Interdisciplinary Coordination**
   - **Risk:** Communication challenges between biological and computational domains
   - **Mitigation:** Maintain clear and regular communication with supervisors from both fields

---

## Literature Review: Han et al. (2020) Analysis

### Study Overview
The eLife paper "Deciphering anomalous heterogeneous intracellular transport with neural networks" presents a machine learning approach to analyze heterogeneous intracellular transport, distinguishing between persistent (directed) and anti-persistent (random/restricted) movement of endosomes and lysosomes.

### Methodology Summary

#### Model Architecture
- **Primary Model:** Deep Learning Feedforward Neural Network (DLFNN)
- **Training Data:** Simulated fractional Brownian motion (fBm) trajectories
- **Architecture Variants:** Triangular, rectangular, and anti-triangular structures
- **Selection Criteria:** Triangular architecture chosen for optimal efficiency

#### Training & Validation Approach
- **Synthetic Data:** fBm trajectories with H values ranging from 0.1 to 0.9
- **Comparison Baseline:** Traditional methods (MSD, rescaled range, sequential range)
- **Robustness Testing:** Short, noisy, and irregular trajectories reflecting real biological data

#### Experimental Application
- **Target Organelles:** Rab5-positive endosomes, SNX1-positive endosomes, lysosomes
- **Analysis Method:** Moving window (15 points) for trajectory segmentation
- **Classification Criteria:**
  - Persistent motion: H > 0.55
  - Anti-persistent motion: H < 0.45
  - Directional classification: anterograde vs. retrograde relative to centrosome

#### Dataset Statistics
The study analyzed 63-71 MRC-5 cells, generating substantial trajectory data:

| Organelle Type | # Trajectories | # Segments (Persistent + Anti-persistent) |
|----------------|----------------|-------------------------------------------|
| GFP-Rab5 Endosomes | 40,800 | 277,926 |
| GFP-SNX1 Endosomes | 11,273 | 215,087 |
| Lysosomes | 38,039 | 474,473 |
| **Total** | **90,112** | **967,486** |

**Note:** Trajectories were segmented using a 15-point moving window approach, resulting in significantly more segments than original trajectories due to overlapping analysis windows.

### Key Findings

#### 1. Superior Performance of DLFNN
- Achieved accurate Hurst exponent estimates with as few as 7 trajectory points
- Outperformed traditional methods, especially on short/noisy data
- Demonstrated robust performance across various data quality conditions

#### 2. Organelle-Specific Behavioral Patterns
- **Rab5 endosomes:** Fastest movement among all organelles
- **SNX1 endosomes:** Exhibited longer persistent phases
- **Lysosomes:** Showed frequent reversals and longest anti-persistent durations
- **Universal Pattern:** All organelles spent more time in anti-persistent states

#### 3. Complex Temporal Dynamics
- Heavy-tailed distributions in both persistent and anti-persistent dwell times
- Suggests complex underlying regulation mechanisms
- Parallels observed in ecological search patterns and human behavior

#### 4. Novel Modeling Approach
- Demonstrated stochastic Hurst exponent interpretation
- Novel model capturing time-varying dynamics in organelle motion

### Limitations & Shortcomings

#### 1. Interpretability Constraints
- Limited biological insight due to deep learning model black-box nature
- Post-hoc visualization required for biological interpretation

#### 2. Biological Validation Gaps
- Functional explanations remain hypothetical
- Motor protein recruitment and actin interactions need experimental validation

#### 3. Model Scope Limitations
- Training exclusively on fBm data
- May miss other transport modes (Lévy walks, confined diffusion)
- Assumes Gaussian noise distribution

#### 4. Generalizability Concerns
- Limited to MRC-5 fibroblast cells
- Unknown performance across different cell types or pathological conditions
- No real-time analysis capability demonstrated

---

## Technical Framework

### Core Concepts

#### Hurst Exponent (H)
- **Definition:** Numerical measure (0-1) characterizing particle movement patterns
- **Interpretation:**
  - H = 0.5: Random walk (Brownian motion)
  - H < 0.5: Anti-persistent movement (direction changes, trapped behavior)
  - H > 0.5: Persistent movement (directional consistency)

#### Fractional Brownian Motion (fBm)
- **Concept:** Mathematical model of random movement with memory
- **Advantage:** Captures temporal correlations absent in pure random walks
- **Application:** Models biological transport more realistically than simple diffusion

#### Neural Network Architectures
- **DLFNN:** Deep Learning Feedforward Neural Network
- **Architecture Types:**
  - **Triangular:** Decreasing neuron count per layer (lightweight, fast)
  - **Rectangular:** Consistent neuron count across layers
  - **Anti-triangular:** Increasing neuron count per layer (memory-intensive)

#### Traditional Analysis Methods
- **MSD (Mean Squared Displacement):** Average displacement over time
- **Rescaled Range:** Measures deviation from average movement
- **Sequential Range:** Movement range over time windows

### Data Processing Pipeline

#### Synthetic Data Generation
1. **fBm Simulation:** Generate trajectories with known H values
2. **Parameter Variation:** Systematic exploration of H and D ranges
3. **Noise Addition:** Controlled introduction of realistic noise levels
4. **Validation Sets:** Multiple datasets for robust model testing

#### Experimental Data Analysis
1. **Trajectory Segmentation:** Moving window approach (15-point segments)
2. **Feature Extraction:** Temporal and spatial characteristics
3. **Classification:** Persistent vs. anti-persistent motion regimes
4. **Directional Analysis:** Anterograde vs. retrograde movement

---

## Critical Analysis & Future Directions

### Current Limitations

#### 1. Simulation-to-Reality Gap
- **Issue:** Training on synthetic fBm data may not capture real biological complexity
- **Concern:** Model may perform well on artificial data but fail on experimental data
- **Risk:** Circular reasoning where model only finds what was built into simulations

#### 2. Biological Relevance
- **Challenge:** Limited validation against known biological mechanisms
- **Gap:** Missing connection between mathematical models and cellular processes
- **Need:** Integration with experimental validation studies

### Proposed Improvements

#### 1. Enhanced Synthetic Data
- **Biological Complexity:** Incorporate motor protein dynamics, organelle interactions
- **Noise Models:** Realistic microscopy artifacts and tracking errors
- **Multi-Modal Transport:** Beyond fBm to include Lévy flights, confined diffusion

#### 2. Advanced Model Architectures
- **Transformer Models:** Better capture long-range temporal dependencies
- **Contrastive Learning:** Unsupervised learning from real trajectory structure
- **Multi-Stage Pipeline:** Separate motion encoding and biological classification

#### 3. Validation Strategies
- **Biological Ground Truth:** Testing on known motor protein mutants
- **Cross-Cell Validation:** Performance across different cell types
- **Real-Time Analysis:** Live imaging capabilities

### Research Impact

#### Scientific Contributions
- **Methodological:** Novel approach to intracellular transport analysis
- **Biological:** Insights into organelle-specific movement patterns
- **Technical:** Robust machine learning framework for trajectory analysis

#### Clinical Relevance
- **Disease Understanding:** Potential insights into neurological disorders
- **Drug Development:** Framework for evaluating therapeutic interventions
- **Diagnostic Tools:** Possible biomarkers for cellular health assessment

---

## Conclusion

This project represents a significant step toward understanding intracellular transport dynamics through machine learning approaches. While current limitations exist in the simulation-to-reality transfer, the framework provides a robust foundation for future research in this critical area of cellular biology.

The integration of advanced neural network architectures with biological validation represents a promising direction for uncovering the complex mechanisms underlying intracellular transport and their implications for human health and disease. 