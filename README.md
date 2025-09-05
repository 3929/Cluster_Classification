<h1 align="center">Hi 👋, I'm Cibi Karthik M</h1>
<h3 align="center">🏭 AI/ML Engineer | Building Smart Solutions for Manufacturing & Process Control</h3>

    
# TargetSuiteCluster
## Overview
A production-grade clustering pipeline for cement mill operational data analysis, featuring automated data processing, clustering, and statistical reporting.

## Project Structure
```
TargetSuiteCluster/
├── run_pipeline.py      # Pipeline execution script
├── FINAL.py            # Core clustering implementation
├── clustering_debug.log # Debug logging output
└── CLUSTER_STATISTICS.xlsx  # Generated statistics
```

## Features

Automated data preprocessing and validation
Configurable clustering parameters
Comprehensive error handling and logging
Multi-silo clustering analysis
Detailed statistical reporting
Excel-based output generation

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy



## Usage
1. Ensure input file `run_pipeline.csv` is present
2. Run the clustering analysis:
```bash
python run_pipeline.py
```
3. Check generated outputs:
   - Dendrogram visualization
   - Cluster distribution plots
   - Feature distribution plots
   - Statistical summary files

## Output Files
- `clustering_debug.log`: Debug logging output
- `CLUSTER_STATISTICS.xlsx`: Detailed cluster statistics

## Visualization
The project generates multiple visualizations:
- Hierarchical clustering dendrogram
- Cluster size distribution
- Feature distributions across clusters

## Analysis Parameters
- Number of clusters: 5
- Clustering method: Ward's linkage
- Data standardization: StandardScaler
- Feature filtering: Automated outlier removal
### 🤝 *Let's Connect!*

<div align="center">
  
<a href="https://linkedin.com/in/cibikarthik" target="_blank">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
</a>
<a href="mailto:cibikarthik.m@gmail.com" target="_blank">
  <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>
</a>
<a href="https://github.com/cibikarthik" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>

</div>
## Contributing
Please follow standard Python coding conventions and document any changes.

## License
Proprietary - All rights reserved
=======
