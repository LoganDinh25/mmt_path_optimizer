# **MMT Path Optimizer**
### *Optimizing Multimodal Transportation Using Nearest Shortest Path*
**Interactive Solver | PuLP/Gurobi + Streamlit | Comprehensive Network Optimization**

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX-green)](https://NetworkX.io/)
[![Gurobi](https://img.shields.io/badge/Optimization-PuLP%252FGurobi-orange)](hhttps://www.gurobi.com/)
[![Streamlit](https://img.shields.io/badge/Web-Streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**MMT Path Optimizer** is a research-driven optimization tool that models and solves **multimodal transportation problems** using the **nearest shortest path** principle using Mixed-Integer Linear Programming (MILP).

It supports real-time exploration of:
- **Multi-modal networks** (road, waterway, etc.)
- **Shortest path computation** with modal switching constraints
- **Cost optimization** across different transport types
- **Multi-commodity** flow optimization (passengers, rice, fish )
- **Visualization** of routes and decisions through a web dashboard

> Built for transportation network planning and infrastructure investment analysis.

---

## Problem Context

The project models a **multi-modal transportation network** where goods and passengers travel across different transport modes.

It seeks to:
- Compute **near-optimal routes** between source and destination nodes  
- Allow **mode-switching** under specific costs  
- Evaluate trade-offs between **distance, time, and mode-change penalties**

---

## Features

| Feature | Description |
|----------|-------------|
| **Nearest Shortest Path Algorithm** | Finds efficient routes considering mode-switch penalties |
| **Interactive Dashboard** | Adjust network parameters in real-time |
| **Optimization** | Solved using Gurobi + PuLP |
| **Visual Network Output** | Plotly-based network visualization |
| **Lightweight Setup** | No Conda, no commercial solvers required |

---

## Demo

Run locally:

```bash
streamlit run transport_optimization_app_.py

Installation (Mac/Linux/Windows – No Conda)
# 1. Install Python 3.9+
# Mac: brew install python
# Ubuntu: sudo apt install python3 python3-pip

# 2. Install dependencies
pip install streamlit pyomo highspy networkx plotly pandas

# 3. Run the app
streamlit run transport_optimization_app_.py


Note: Works entirely with open-source solvers — no Gurobi or Conda required.

Project Structure
mmt_path_optimizer/
│
├── transport_optimization_app_.py                   # Streamlit web app
├── model_core.py             # Pmodel for nearest-shortest-path
├── utils/                    # Helper modules (graph building, plotting)
├── data/                     # Input CSVs (nodes, arcs, parameters)
├── requirements.txt
├── README.md
└── assets/
    └── network_example.png

Model Overview
Decision Variables

x[i,j,m] — flow along arc (i,j) using mode m

y[m] — binary variable: whether mode m is used

z[i,j] — binary: whether arc (i,j) is chosen

Objective

Minimize:

Σ(distance[i,j] * x[i,j,m]) + Σ(mode_switch_cost * y[m])

Constraints

Flow conservation

Mode-switch limits

Arc feasibility and capacity

Nonnegative variables

Visualization

Hover to explore shortest paths and mode changes.

Future Work

Integration with LLMs for route explanation in natural language

Support for dynamic travel times and period-based demand

Cloud-based solver interface (via API)

References

 [1]Fragkos, I., Cordeau, J. F., & Jans, R., “The Multi-Period Multi-Commodity Network Design Problem,” CIRRELT Technical Report, 2017.

[2] Jernigan, N. R., Multi-Modal, Multi-Period, Multi-Commodity Transportation: Models and Algorithms, Diss. Massachusetts Institute of Technology, 2014.

[3] Orozco-Fontalvo, M., Cantillo, V., & Miranda, P. A., “A Stochastic, Multi-Commodity Multi-Period
Inventory-Location Problem: Modeling and Solving an Industrial Application,” in Proceedings of the
International Conference on Computational Logistics, pp. 317–331, Cham: Springer International
Publishing, 2019.

[4] Bayram, V., Aydo˘gan, C¸ . & Kargar, K., ”Multi-period hub network design from a dual perspective:
An integrated approach considering congestion, demand uncertainty, and service quality optimization”, European Journal of Operational Research, 2025.


License

MIT License
 – Free to use and modify.

Author

Đinh Xuân Trường
University of Kent
Email: truongdinh4w@gmail.com