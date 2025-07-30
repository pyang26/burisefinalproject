# Brain-Inspired Reinforcement Learning Model

A dopamine-centered reinforcement learning model that simulates how dopamine modulates reward processing and learning in the human brain.

## 🧠 Overview

This project implements a **dopamine-centered reinforcement learning model** where dopamine is the ONLY input. The model simulates how different dopamine levels affect Reward Prediction Error (RPE) computation and learning, providing insights into brain reward processing mechanisms.

### **Dopamine Scenarios**
1. **HIGH_DOPAMINE** (0.8+) → Enhanced learning (reward-rich environments)
2. **LOW_DOPAMINE** (0.2-) → Suppressed learning (depression-like states)
3. **NORMAL_DOPAMINE** (0.5) → Baseline learning (healthy brain)
4. **VARIABLE_DOPAMINE** (0.3-0.7) → Unpredictable learning (addiction-like)
5. **DEPLETED_DOPAMINE** (0.1-) → Impaired learning (Parkinson's-like)

### **Core Components**
- **DopamineCenteredRPE**: RPE computation with dopamine modulation
- **DopamineCenteredAgent**: Agent that learns based on dopamine input
- **Modulation Network**: Learns how dopamine affects RPE computation
- **Value Network**: Predicts future rewards

## 📊 Model Architecture

```
Dopamine Input → Modulation Network → RPE Computation
     ↓              ↓                    ↓
Dopamine Level → Modulation Factor → Modulated RPE
```

### **RPE Computation**
Base RPE = reward + γ × V(next_state) - V(current_state)
Modulated RPE = Base RPE × (1 + dopamine_modulation_factor)
### **Dopamine Modulation**
- **Higher dopamine** → **Enhanced RPE** (amplified learning)
- **Lower dopamine** → **Suppressed RPE** (attenuated learning)
- **Modulation factor** learned by neural network

### **High Dopamine Scenarios**
- **Enhanced learning** and reward processing
- **Amplified RPE** computation
- **Positive modulation factors**
- **Similar to reward-rich environments**

### **Low Dopamine Scenarios**
- **Suppressed learning** and reward processing
- **Attenuated RPE** computation
- **Negative modulation factors**
- **Similar to depression or anhedonia**

### **Depleted Dopamine Scenarios**
- **Severely impaired learning**
- **Strong negative modulation**
- **Minimal RPE computation**
- **Similar to Parkinson's disease**

### **Key Insights**
- **Dopamine acts as a neuromodulator** for RPE computation
- **Higher dopamine** → **Enhanced learning** (positive modulation)
- **Lower dopamine** → **Suppressed learning** (negative modulation)
- **Modulation factors** learned automatically by the network

### **Biological Mechanisms**
1. **Dopamine Release**: VTA neurons release dopamine in response to rewards
2. **RPE Computation**: Brain computes reward prediction errors
3. **Learning Modulation**: Dopamine modulates how RPE affects learning
4. **Value Updates**: RPE drives updates to reward predictions

### **Dependencies**
- `torch`: Neural network implementation
- `numpy`: Numerical computations
- `matplotlib`: Visualization and plotting

### **Model Components**
- **State Encoder**: Processes environment state
- **Value Network**: Predicts future rewards
- **Modulation Network**: Learns dopamine effects
- **RPE Computer**: Computes reward prediction errors

### **Learning Algorithm**
- **Experience Replay**: Stores and learns from past experiences
- **Neural Network Updates**: Updates based on RPE
- **Dopamine Modulation**: Adjusts learning based on dopamine levels
