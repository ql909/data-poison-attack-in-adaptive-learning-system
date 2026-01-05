# Overview

This repository contains supplementary materials for the following conference paper:

Qinyi Liu, Lin Li, Valdemar Švábenský, Conrad Borchers, and Mohammad Khalil.\
**Measuring the Impact of Student Gaming Behaviors on Learner Modeling**.\
In Proceedings of the 16th International Learning Analytics and Knowledge Conference (LAK '26).\
https://doi.org/10.1145/3785022.3785036

```bibtex
@inproceedings{Liu2026measuring,
    author    = {Liu, Qinyi and Li, Lin and \v{S}v\'{a}bensk\'{y}, Valdemar and Borchers, Conrad and Khalil, Mohammad},
    title     = {{Measuring the Impact of Student Gaming Behaviors on Learner Modeling}},
    booktitle = {Proceedings of the 16th International Learning Analytics and Knowledge Conference},
    series    = {LAK '26},
    location  = {Bergen, Norway},
    publisher = {Association for Computing Machinery},
    year      = {2026},
    numpages  = {11},
    url       = {https://doi.org/10.1145/3785022.3785036},
    doi       = {10.1145/3785022.3785036},
}
```

## Folders

- `AFM/`: Contains scripts and configurations for Additive Factor Model (AFM) model.
- `BKT/`: Contains scripts and configurations for Bayesian Knowledge Tracing (BKT) model. 
- `DKT/`: Contains scripts and configurations for Deep Knowledge Tracing (DKT) model. 
- `dataset_split/`: Houses scripts and utilities for splitting datasets at a student level and skills levels.
- `dpa_data_generation/`: Includes code and simulation gaming behaviors with DPA techniques.

## AUC results

<div align="center">
  
### AUC values on the test set for student splitting
</div>

| Dataset       | KT-Model | Raw   | Random 5 | Random 25 | Random 50 | SeqPat 5 | SeqPat 25 | SeqPat 50 | Hint 5 | Hint 25 | Hint 50 |
| ------------- | -------- | ----- | -------- | --------- | --------- | -------- | --------- | --------- | ------ | ------- | ------- |
| **Dataset B** | BKT      | 0.638 | 0.631    | 0.635     | 0.639     | 0.635    | 0.639     | 0.632     | -      | -       | -       |
|               | DKT      | 0.882 | 0.875    | 0.881     | 0.883     | 0.881    | 0.883     | 0.878     | -      | -       | -       |
|               | AFM      | 0.790 | 0.712    | 0.789     | 0.789     | 0.789    | 0.789     | 0.789     | -      | -       | -       |
| **Dataset A** | BKT      | 0.603 | 0.533    | 0.605     | 0.602     | 0.605    | 0.602     | 0.603     | 0.608  | 0.589   | 0.601   |
|               | DKT      | 0.753 | 0.616    | 0.751     | 0.753     | 0.751    | 0.753     | 0.752     | 0.752  | 0.751   | 0.748   |
|               | AFM      | 0.570 | 0.527    | 0.571     | 0.571     | 0.571    | 0.571     | 0.572     | 0.571  | 0.577   | 0.575   |

<div align="center">

### AUC values on the test set for skill splitting

</div>


| Dataset       | KT-Model | Raw    | Random 5 | Random 25 | Random 50 | SeqPat 5 | SeqPat 25 | SeqPat 50 | Hint 5 | Hint 25 | Hint 50 |
| ------------- | -------- | ------ | -------- | --------- | --------- | -------- | --------- | --------- | ------ | ------- | ------- |
| **Dataset B** | BKT      | 0.637  | 0.620    | 0.625     | 0.600     | 0.625    | 0.600     | 0.640     | -      | -       | -       |
|               | DKT      | 0.846  | 0.830    | 0.848     | 0.846     | 0.848    | 0.846     | 0.847     | -      | -       | -       |
|               | AFM      | 0.755  | 0.543    | 0.7549    | 0.7549    | 0.7549   | 0.7549    | 0.7549    | -      | -       | -       |
| **Dataset A** | BKT      | 0.573  | 0.525    | 0.573     | 0.571     | 0.573    | 0.571     | 0.574     | 0.572  | 0.560   | 0.554   |
|               | DKT      | 0.710  | 0.541    | 0.711     | 0.708     | 0.711    | 0.708     | 0.711     | 0.709  | 0.704   | 0.693   |
|               | AFM      | 0.6471 | 0.5089   | 0.6138    | 0.6136    | 0.6138   | 0.6136    | 0.6137    | 0.6029 | 0.5970  | 0.5914  |



## KT models math formula

### Additive Factor Model (AFM)

<p align="center">

$$
P(Y_{i,j} = 1) = \sigma\!\big( \theta_i + \beta_{k(j)} + \gamma_{k(j)} N_{i,k(j)} \big)
$$

</p>

**Explanations**  
- \( Y_{i,j} \): correctness of student \(i\) on item \(j\) (1 = correct, 0 = incorrect)  
- \( \sigma(x) = \frac{1}{1+e^{-x}} \): sigmoid function  
- \( \theta_i \): proficiency of student \(i\)  
- \( \beta_{k(j)} \): difficulty of skill \(k\) associated with item \(j\)  
- \( \gamma_{k(j)} \): learning rate of skill \(k\)  
- \( N_{i,k(j)} \): number of prior opportunities student \(i\) has practiced skill \(k\)  

**Reference**  
Cen, H., Koedinger, K., & Junker, B. (2006). *Learning factors analysis – A general method for cognitive model evaluation and improvement.* In **Intelligent Tutoring Systems (ITS)**.  

---

### Bayesian Knowledge Tracing (BKT)

<p align="center">

$$
P(C_t) = P(L_{t-1})(1-S) + (1-P(L_{t-1}))G
$$  

$$
P(L_t) = \frac{ P(L_{t-1})(1-S) }{P(C_t)} \quad \text{(if correct)}
$$  

$$
P(L_t) = \frac{ P(L_{t-1})S }{1-P(C_t)} \quad \text{(if incorrect)}
$$  

$$
P(L_t) \leftarrow P(L_t) + (1-P(L_t))T
$$  

</p>

**Explanations**  
- \( P(L_0) \): initial probability of mastery  
- \( P(T) \): transition probability (learning: unmastered → mastered)  
- \( P(S) \): slip probability (mastered but answered incorrectly)  
- \( P(G) \): guess probability (unmastered but answered correctly)  
- \( P(C_t) \): probability of a correct response at time \(t\)  
- \( P(L_t) \): probability of mastery at time \(t\)  

**Reference**  
Corbett, A. T., & Anderson, J. R. (1995). *Knowledge tracing: Modeling the acquisition of procedural knowledge.* **User Modeling and User-Adapted Interaction, 4(4), 253–278.**

---

### Deep Knowledge Tracing (DKT)

<p align="center">

$$
h_t = f(W_{hx} x_t + W_{hh} h_{t-1} + b_h)
$$  

$$
\hat{y}_{t+1} = \sigma(W_{yh} h_t + b_y)
$$  

</p>

**Explanations**  
- \( x_t \): input vector representing student interaction (question + response) at time \(t\)  
- \( h_t \): hidden state representing student knowledge at time \(t\)  
- \( f \): RNN cell function (e.g., tanh, LSTM, or GRU)  
- \( \hat{y}_{t+1} \): predicted probability of a correct response at time \(t+1\)  

**Reference**  
Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J. (2015). *Deep knowledge tracing.* In **Advances in Neural Information Processing Systems (NeurIPS)**.  


