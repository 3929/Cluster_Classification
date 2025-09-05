<!-- Center and highlight the Optiverse logo in README.md -->

<div align="center" style="background: #f0f8ff; padding: 24px; border-radius: 16px; box-shadow: 0 2px 8px #e0e0e0;">
  <!-- You can adjust the background color and padding as needed -->
  <img src="https://github.com/user-attachments/assets/35bef16e-453c-4fdd-b362-5ced35d19898" alt="Optiverse Logo" width="350"/>
  <h2 style="color: #01B763; margin-top: 16px;">
</div>
    
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

## Contributing
Please follow standard Python coding conventions and document any changes.

## License
Proprietary - All rights reserved
=======
