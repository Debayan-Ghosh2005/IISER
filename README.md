<h1 align="center">🧠 Machine Learning Fine-Tuning & Reinforcement Learning Project</h1>

<div align="center">
  <strong>Compare Base, Fully Fine-Tuned, K/Few/One-Shot Models & Begin Reinforcement Learning Journey</strong><br/>
  📁 Based on IMDB Dataset | 🧪 Built with Scikit-learn | 🤖 Extending to Reinforcement Learning | 👨‍💻 Author: <b>Debayan Ghosh</b>
</div>

---

## 📌 Overview

This project demonstrates how different machine learning fine-tuning strategies impact performance on a classification problem. It includes:

- ✅ **Baseline Training**
- 🔄 **Full Fine-Tuning**
- 🎯 **Few-Shot & One-Shot Fine-Tuning**
- 📊 **Threshold Tuning for F1 Optimization**
- 🚀 **Beginning of Reinforcement Learning Integration**

The goal is to simulate real-world scenarios where labeled data is limited and explore reinforcement learning techniques in upcoming stages.

---

## 📂 Project Structure

```bash
D:.
├── README.md                              # Main project README

├── Classification/
│   ├── F1.py                              # F1 score evaluator for predictions
│   ├── finetune.py                        # Fully fine-tuned model
│   ├── IMDB.csv                           # Original IMDB dataset (full version)
│   └── zeroandfewshot.py                  # Zero-shot & few-shot classification

├── Fine tune/
│   ├── comparison_predictions.csv         # Logs predictions for analysis
│   ├── data.csv                           # Processed dataset (1000 rows)
│   ├── data1.csv                          # Additional or backup dataset
│   ├── F1.py                              # Threshold-tuned F1 scoring
│   ├── finetune.py                        # RandomizedSearchCV-based tuning
│   ├── kshort.py                          # K-shot learning logic
│   └── oneshot.py                         # One-shot learning implementation

└── Reinforcement_Learning/
    ├── rl_agent.py                        # First RL model (Q-Learning/Policy Gradient)
    ├── env_setup.py                       # Environment loading and preprocessing
    └── README.md                          # RL-specific instructions

---

## 💡 Features

- 📈 Evaluate and compare different fine-tuning strategies  
- ⚙️ Implements threshold optimization for best F1 score  
- 🔍 Simulates low-label scenarios (few-shot learning)  
- 🔄 Adds modular support for extension to RL models  
- 🧪 Scikit-learn powered pipeline, easy to extend  

---

## 🔁 Upcoming: Reinforcement Learning (RL)

We're beginning the next phase with **Reinforcement Learning**, aimed to include:

- 🧠 Basic agent-environment interaction  
- 🕹️ Simple OpenAI Gym environments (e.g., CartPole, FrozenLake)  
- 📊 Comparing supervised vs. RL-based decision making  
- 🔄 Future script additions: `rl_agent.py`, `env_setup.py`

---

## 🛠️ Tech Stack

| Technology       | Description                    |
|------------------|--------------------------------|
| **Python**       | Core Programming Language       |
| **Pandas**       | Data Manipulation               |
| **NumPy**        | Numerical Operations            |
| **Scikit-learn** | ML Models & Evaluations         |
| **Matplotlib**   | (Upcoming) Visualizations       |
| **OpenAI Gym**   | (Upcoming) RL Environments      |

---

## 🚀 How to Run

1. **Install required packages**  
```bash
pip install pandas numpy scikit-learn
---

## 👨‍💻 Author

**Debayan Ghosh**  
🎓 B.Tech in Computer Science (AI/ML) @ MCKV Institute of Engineering  
📚 Online BSc Data Science @ IIT Madras  
🌐 [GitHub](https://github.com/Debayan-Ghosh2005) • [LinkedIn](https://www.linkedin.com/in/myself-debayan-ghosh/)

---

## 📜 License

This project is for **educational and research purposes only**.  
Feel free to fork or adapt the scripts for learning.
