
# 🧠 MEMS-AgentLab: Multi-Model Physics-Informed Neural Networks for Micro-Scale Resonator Systems

### *Agent-Orchestrated PINNs for Parameter Inversion and Rapid Simulation*

---

## 🚀 Project Overview

**MEMS-AgentLab** aims to establish a new paradigm for **multi-model orchestration in Physics-Informed Neural Networks (PINNs)**.
Taking the **MEMS microbeam resonator** as a canonical example, this project explores how **multiple specialized PINN models**—each trained for a constrained subset of parameters—can collectively outperform a single over-generalized network.

Instead of building one “super-model” to handle all possible parameter combinations (which scales exponentially in complexity),
this framework proposes a **lightweight memory-rich approach**:

> “💡 Memory is cheaper than computation — so let’s train many specific models, and let an intelligent agent decide which one to use.”

The project envisions an **Agent-Mapping System** that:

1. **Interprets user queries or simulation requests** (e.g., “Given Q = 10⁴ and V = 5 mV, predict wₜ, lₜ.”)
2. **Matches the query to the most suitable PINN model** within the model repository
3. **Executes rapid forward inference**, returning physics-consistent results
4. **Learns from usage data**, continuously refining model-mapping accuracy

This design allows researchers and engineers to **interact with PINN-driven solvers just like traditional simulation tools**,
but with **orders-of-magnitude faster response**, minimal numerical iteration, and embedded physical interpretability.

---

## 🧩 Scientific Background

The foundation of this project originates from prior work:
**“PINN-based Parameter Inversion for Microbeam Resonators”**.

A MEMS micro-cantilever resonator can be modeled as a **nonlinear spring–mass–damper system**:

$$
m \frac{d^2x}{dt^2} + c \frac{dx}{dt} + kx = F(t)
$$

At the microscale, the system exhibits strong **geometric nonlinearity**, **surface effects**, and **electromechanical coupling**,
requiring a set of **nonlinear algebraic and PDE constraints**:

$$
\begin{cases}
-M\omega^2 y + (k_t - k_e)y + (k_{t3} - k_{e3})\frac{3}{4}y^3 - F_{\text{ac}}\cos(\phi) = 0 \
c\omega y - F_{\text{ac}}\sin(\phi) = 0
\end{cases}
$$

These physical relations provide the **governing residuals** for the PINN framework, enabling the inversion of structural parameters
such as **beam width ($w_t$)**, **beam length ($l_t$)**, **quality factor (Q)**, and **actuation voltage (V)**.

---

## 🧠 Motivation

* Training a single PINN to infer all parameters simultaneously faces **exponential complexity** with dimensionality.
* Training independent networks for each subset of parameters yields **narrow specialization but poor generalization**.
* Therefore, the project adopts a **multi-model intelligence strategy**:

  * Each model specializes in a well-defined parameter regime.
  * A meta-agent maintains a **mapping graph** from parameter space to model index.
  * Upon receiving a user query, the agent performs model selection and inference composition.

This effectively transforms the problem from

> “building one model to solve everything”
> into
> “building a knowledge network of models, coordinated by an intelligent agent.”

---

## 🧮 Core Components

| Module              | Description                                                            |
| :------------------ | :--------------------------------------------------------------------- |
| **PINN-QFactor**    | Infers the quality factor (Q) given geometry and spectral data         |
| **PINN-wₜlₜ**       | Inverts geometric parameters (wₜ, lₜ) from frequency response          |
| **Agent-Mapper**    | Learns the mapping between parameter regimes and model indices         |
| **Knowledge Graph** | Stores metadata (parameter ranges, residual forms, error metrics)      |
| **Interface Layer** | Provides API/CLI for users to query physics models via natural prompts |

---

## 🔬 Current Progress

* ✅ Reconstructed physics-consistent PINN frameworks for `Q` and `(wₜ, lₜ)` inversion
* ✅ Established dataset generation pipeline with HDF5 storage and MinMax normalization
* ✅ Implemented visualization suite (error maps, 3D animations, rotation GIFs)
* 🔄 Designing **Agent-Mapping layer** for model selection and adaptive routing
* 🔜 Future: Online learning for agent behavior and real-time user interaction

---

## 🌍 Vision

The long-term goal of MEMS-AgentLab is to build an **intelligent multi-model inference system** for physics-based simulation tasks.
In the future, users can input problems as naturally as they do in traditional solvers (e.g., COMSOL or ANSYS),
while the agent internally selects appropriate PINN models and returns solutions in milliseconds.

---


