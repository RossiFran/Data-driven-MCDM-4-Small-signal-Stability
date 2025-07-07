# Data-driven MCDM algorithm
The workflow of the data-driven MCDM algorithm can be summarized as in the figure below.
<img src="https://github.com/user-attachments/assets/7d715085-a025-40c2-aadd-7116d0026d99" alt="decision_alg" width="400"/>

Consider that the system has to transit from the operating point (OP) marked with a full circle (•) to another, marked with an empty circle (◦).
The MCDM problem consists of finding, among all the possible alternatives, the CCRC to be assigned at OP ◦ (i.e., X◦C ) that makes the system stable and with
the best stability performances, according to multiple indicators. Concerning the online implementation of the algorithm, this goal can be achieved thanks to
the use of data-driven surrogate models for the computation of the stability and the stability performance indicators. However, the speed-up in computation is
paid for in terms of accuracy. Therefore, before operating the system with the proposed X◦C , a verification of the stability response through the exact models
is carried out. Hence, as depicted in the figure, the main steps of the algorithm are the data-driven MCDM and the verification of the data-driven solution.
Next, the pseudocode describes the instructions implemented in the main steps, as performed in the code `data_driven_decision_making_algorithm.m`.

<img src="https://github.com/user-attachments/assets/e0bddc9f-81bd-4f51-9f96-961b6a7f032b" alt="GoG_algorithm" width="600"/>
