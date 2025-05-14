# Slime-Mould-Algorithm-for-Breast-Cancer-Disease-Subtype-Classification-Using-Multiomics-Data-
#This project presents an intelligent web-based tool for classifying breast cancer subtypes based on multiomics data using advanced optimization algorithms. The core objective is to enhance classification performance through Slime Mould Algorithm (SMA)-based feature selection, and compare its performance against Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) using Random Forest and SVM classifiers.
#Project Structure
BCD Care/
│
├── app.py                           # Main Flask application
│
├── models/                          # Model scripts and saved .pkl artifacts
│   ├── randomforest_withsma.py
│   ├── randomforest_withga.py
│   ├── randomforest_withpso.py
│   ├── svm_withsma.py
│   ├── svm_withga.py
│   ├── svm_withpso.py
│   ├── *.pkl                        # Trained models and selected feature files
│   └── report.log                   # Logging information
│
├── project/uploads/                # Dataset storage
│   └── brca_data_w_subtypes (1).csv
│
├── static/
│   ├── *.png                        # Logos and image assets
│   ├── css/
│   │   ├── detection.css
│   │   ├── eda.css
│   │   ├── navbar.css
│   │   ├── resources.css
│   │   └── style.css
│   └── js/
│       ├── detection.js
│       └── eda.js
│
├── templates/
│   ├── index.html
│   ├── detection.html
│   ├── eda.html
│   └── resources.html
│
└── uploads/
    └── brca_data_w_subtypes (1).csv  # Duplicate for runtime
