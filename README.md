# Data-driven-MCDM-4-Small-signal-Stability
This repository contains the code for the paper Data-Driven Decision Making for Enhancing Small-Signal Stability in Hybrid AC/DC Grids Through Converter Control Role Assignment[1]. The

Hybrid AC/DC Transmission Grids integrate High-Voltage Direct Current (HVDC) links through Interconnecting Power Converters (IPCs). The control role assigned to each IPC plays a key role in shaping grid dynamics. While traditional IPCs operate with fixed control roles, recent research has explored day-ahead scheduling of these roles to improve system stability. However, in power systems with high renewable energy penetration, forecast inaccuracies can make pre-scheduled control assignments suboptimal—or even destabilizing.

To overcome this limitation, this work introduces an online scheduling recalculation algorithm that dynamically updates IPC control roles during real-time operation. The algorithm is based on a data-driven, multi-criteria decision-making framework that incorporates surrogate models of conventional small-signal stability analysis tools. This allows for a rapid evaluation of system stability and performance metrics.

The proposed methodology is demonstrated on the test system shown in the Figure below [2].
This repository contains all the necessary code to implement and reproduce the methodology for this case study.
<br>
<br>
![power_syst_BigModel_bus.pdf](https://github.com/user-attachments/files/21076031/power_syst_BigModel_bus.pdf)
<br>
<br>

### 📁 Repository Structure

#### `CCRCs_clustering/`

Contains all scripts required to perform **clustering of the Candidate Converter Role Configurations (CCRCs)** based on their dynamic behavior with respect to each stability performance indicator.
The methodology identifies groups of CCRCs exhibiting similar stability characteristics, and then applies a **set intersection technique** to determine the optimal subset of CCRCs. This subset ensures robust dynamic performance across the entire operating space according to multiple stability indicators.

#### `Datasets/`
Collects:
* The full dataset of exact small-signal stability assessment results computed for all CCRCs

* Scripts for data cleaning and feature engineering, including:

  * Conversion to per-unit (p.u.) values

  * Creation of additional features

  * Removal of highly correlated variables
  *  Reusable data cleaning functions to be applied to new datasets before training or testing the models

#### `Training_stability_assessment/`

Includes the code for training **data-driven surrogate models** that approximate the exact small-signal stability assessment.
The training workflow comprises:

* Data cleaning
* Feature engineering
* Initial model screening
* Feature selection using **permutation feature importance**
* Hyperparameter tuning via **grid search with k-fold cross-validation**
* Final model training

#### `Training_indicators_regression/`

Contains the scripts to train surrogate models that estimate the **quantitative values of small-signal stability performance indicators**.
The training pipeline follows a similar structure:

* Data cleaning
* Feature engineering
* Initial model screening
* Feature selection using **permutation feature importance**
* Final model training






## References
[1] Rossi, Francesca, et al. "Data-Driven Decision Making for Enhancing Small-Signal Stability in Hybrid AC/DC Grids Through Converter Control Role Assignment." arXiv preprint arXiv:2503.05386 (2025).

[2] J. A. Soler, D. Groß, E. P. Araujo, O. G. Bellmunt, Interconnecting power converter control role assignment in grids with multiple ac and dc subgrids, IEEE Trans. Power Del. 38 (3) (2023). doi:10.1109/TPWRD.2023.3236977.
