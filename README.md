# Slime-Mould-Algorithm-for-Breast-Cancer-Disease-Subtype-Classification-Using-Multiomics-Data-
This project presents an intelligent web-based tool for classifying breast cancer subtypes based on multiomics data using advanced optimization algorithms. The core objective is to enhance classification performance through Slime Mould Algorithm (SMA)-based feature selection, and compare its performance against Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) using Random Forest and SVM classifiers.




## 📁 Project Structure

<pre>
<code>
📦 BCD Care/
├── app.py                          # 🧠 Main Flask application
├── models/                         # 🤖 ML model scripts & saved artifacts
│   ├── randomforest_withsma.py     # RF + Slime Mould Algorithm
│   ├── randomforest_withga.py      # RF + Genetic Algorithm
│   ├── randomforest_withpso.py     # RF + Particle Swarm Optimization
│   ├── svm_withsma.py              # SVM + Slime Mould Algorithm
│   ├── svm_withga.py               # SVM + Genetic Algorithm
│   ├── svm_withpso.py              # SVM + Particle Swarm Optimization
│   ├── *.pkl                       # 📦 Trained model & feature selection files
│   └── report.log                  # 📝 Model training logs
├── project/uploads/                # 📂 Raw dataset storage
│   └── brca_data_w_subtypes (1).csv
├── static/                         # 🎨 Front-end assets
│   ├── *.png                       # Logos & images
│   ├── css/                        # CSS styles
│   │   ├── detection.css
│   │   ├── eda.css
│   │   ├── navbar.css
│   │   ├── resources.css
│   │   └── style.css
│   └── js/                         # JavaScript files
│       ├── detection.js
│       └── eda.js
├── templates/                      # 🧾 HTML templates
│   ├── index.html                  # Home page
│   ├── detection.html              # Detection module UI
│   ├── eda.html                    # Exploratory Data Analysis page
│   └── resources.html              # Resource links & documents
└── uploads/                        # 📥 User-uploaded CSVs (runtime)
    └── brca_data_w_subtypes (1).csv
</code>
</pre>
