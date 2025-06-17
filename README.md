# TCMPrime

**TCMprime: Node-Weighted Network Pharmacology for TCM Formula Optimization**

Our method explicitly integrates protein node importance and herb-specific data to prioritize TCM formula components, identify bioactive compounds, and clarify mechanisms of action — illustrated with a case study on Parkinson’s disease.

---

## Requirements

- networkx  
- numpy  
- scipy  
- pandas  
- jupyter  
- matplotlib  

---

## Code Overview

**herb-compound-target-process.ipynb**  
This Jupyter notebook provides data processing and weight calculation for herb-compound-target relationships.  
It implements and demonstrates three weighting strategies:  
- **Method 1:** Herb proportion only  
- **Method 2:** Herb proportion + compound abundance  
- **Method 3:** Herb proportion + compound abundance + compound–target interaction probabilities  
The notebook outputs normalized target weights for downstream network analysis.  

**network-analysis/**  
This folder contains Python scripts and utilities for performing the main network pharmacology computations.  
Functions include:  
- Construction of protein–protein interaction networks  
- Calculation of target coverage, Jaccard similarity, network proximity (weighted/unweighted)  
- Statistical significance evaluation (Z-scores, random sampling)  

