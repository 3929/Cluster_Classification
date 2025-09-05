from sklearn.cluster import AgglomerativeClustering as Agglomerative
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import scipy.cluster.hierarchy as shc
from typing import Dict, List, Any, Union
import os
import warnings
import logging
from datetime import timedelta, datetime
from logging.handlers import TimedRotatingFileHandler

warnings.filterwarnings('ignore')

# Configure logging with file rotation every 24 hours
def setup_logging():
    """
    Setup logging configuration with both console and file handlers.
    File logs are rotated every 24 hours.
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with daily rotation
    log_file = os.path.join(logs_dir, 'clustering_pipeline.log')
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',  # Rotate at midnight
        interval=1,       # Every 1 day
        backupCount=30,   # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Set the suffix for rotated files (YYYY-MM-DD format)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.namer = lambda name: name.replace(".log", "") + ".log"
    
    logger.addHandler(file_handler)
    
    return logger

# Setup logging
logger = setup_logging()

class CLUSTER:
    def __init__(self):
        """
        Initialize:
        - Database connections
        - Configuration parameters
        - Logging setup
        """
        pass
 
    def segregate_silo_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segregate data by SILO.
       
        Args:
            df (pd.DataFrame): Input data
           
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with Silo names as keys and corresponding DataFrames as values
        """
        try:
            # First identify and standardize the datetime column
            datetime_cols = df.select_dtypes(include=['datetime64', 'object']).columns
            detected_datetime = None
            
            for col in datetime_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().sum() > 0:
                        detected_datetime = col
                        break
                except Exception:
                    continue
            
            if not detected_datetime:
                raise ValueError("No valid datetime column found in DataFrame")
            
            logger.info(f"Detected datetime column: '{detected_datetime}'")
            if detected_datetime != 'DateTime':
                df = df.rename(columns={detected_datetime: 'DateTime'})

            silo_data = {}
            silo_columns = [col for col in df.columns if col.startswith('Silo')]
           
            if not silo_columns:
                raise ValueError("No silo selection columns found in the data")
           
            # For each row, determine which silo is selected (value 1)
            df['SILO'] = 'Unknown'
            for idx, row in df.iterrows():
                for silo_col in silo_columns:
                    if row[silo_col] == 1:
                        silo_num = silo_col.replace('Silo', '').replace('Sel', '')
                        df.at[idx, 'SILO'] = f'Silo{silo_num}'
           
            # Now segregate by the determined SILO
            for silo in df['SILO'].unique():
                if silo == 'Unknown':
                    continue
                silo_df = df[df['SILO'] == silo].copy()
                if not silo_df.empty:
                    silo_data[silo] = silo_df
                    logger.info(f"Processing {len(silo_df)} records for {silo} Segregation...")
           
            if not silo_data:
                raise ValueError("No valid silo data found after segregation")
               
            return silo_data
           
        except Exception as e:
            logger.error(f"Error in segregating silo data: {str(e)}")
            raise

    def process_silo_data(self, silo_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        try:
            processed_data = {}
           
            for silo, data in silo_data.items():
                # Log initial data details
                logger.info(f"Processing {silo}: Initial records = {len(data)}")
                logger.info(f"DateTime range: {data['DateTime'].min()} to {data['DateTime'].max()}")
               
                # Ensure the DataFrame is sorted by DateTime
                df_sorted = data.sort_values('DateTime').reset_index(drop=True)
               
                # Calculate time differences
                df_sorted['time_diff'] = df_sorted['DateTime'].diff().dt.total_seconds() / 3600
               
                # Logging time difference statistics
                logger.info(f"Time difference statistics for {silo}:")
                logger.info(f"  Min time diff: {df_sorted['time_diff'].min()} hours")
                logger.info(f"  Max time diff: {df_sorted['time_diff'].max()} hours")
               
                # Modify filtering conditions
                # 1. Only remove rows with extremely large time gaps (e.g., > 24 hours)
                df_filtered = df_sorted[
                    df_sorted['time_diff'].isna() |  # Keep first row
                    (df_sorted['time_diff'] <= 24)   # Keep rows with time diff <= 24 hours
                ]
               
                # 2. Optionally, keep at least a minimum number of records
                if len(df_filtered) >= 10:  # Adjust this threshold as needed
                    # Drop the temporary time_diff column
                    df_processed = df_filtered.drop(columns=['time_diff']).reset_index(drop=True)
                   
                    # Store processed data
                    processed_data[silo] = df_processed
                   
                    logger.info(f"Processed {silo}: Removed {len(data) - len(df_processed)} rows")
                    logger.info(f"Remaining records: {len(df_processed)}")
                else:
                    logger.warning(f"{silo} has insufficient records after filtering: {len(df_filtered)} records")
           
            return processed_data
       
        except Exception as e:
            logger.error(f"Error in processing silo data: {str(e)}")
            raise

    def extract_and_convert_hourly(self, df: pd.DataFrame, quality_parameter: str) -> pd.DataFrame:
        """
        Process the dataset according to the following rules:
        1. Sort data by time in ascending order
        2. Extract parameter values with their time ranges
        3. Skip first 15 mins of each parameter value period and average remaining data
        4. Remove parameter periods less than 45 mins
        5. Shift averages up one step relative to parameter values
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw data
            quality_parameter (str): Column name of the quality parameter to process
        
        Returns:
            pd.DataFrame: Processed dataframe with shifted averages
        """
        try:
            # Validate inputs
            if quality_parameter not in df.columns:
                raise ValueError(f"Quality parameter '{quality_parameter}' not found in DataFrame")

            # Auto-detect the datetime column
            datetime_cols = df.select_dtypes(include=['datetime64', 'object']).columns
            detected_datetime = None
            
            for col in datetime_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().sum() > 0:
                        detected_datetime = col
                        break
                except Exception:
                    continue
            
            if not detected_datetime:
                raise ValueError("No valid datetime column found in DataFrame")
            
            logger.info(f"Detected datetime column: '{detected_datetime}'")
            df.rename(columns={detected_datetime: 'DateTime'}, inplace=True)

            # 1. Sort by date in ascending order (oldest to latest)
            df = df.dropna(subset=['DateTime']).sort_values('DateTime', ascending=True).reset_index(drop=True)
            logger.info("Sorted data in ascending order by time (oldest to latest)")
            
            # 2. Extract parameter value periods
            periods = []
            current_value = df[quality_parameter].iloc[0]
            period_start = df['DateTime'].iloc[0]
            
            logger.info(f"Extracting {quality_parameter} value periods...")
            for idx in range(1, len(df)):
                if df[quality_parameter].iloc[idx] != current_value:
                    period_end = df['DateTime'].iloc[idx-1]
                    duration = period_end - period_start
                    
                    periods.append({
                        'parameter_value': current_value,
                        'from_date': period_start,
                        'to_date': period_end,
                        'duration_minutes': duration.total_seconds() / 60,
                        'start_idx': idx-1,
                        'end_idx': idx-1
                    })
                    
                    current_value = df[quality_parameter].iloc[idx]
                    period_start = df['DateTime'].iloc[idx]
            
            # Add the last period
            period_end = df['DateTime'].iloc[-1]
            duration = period_end - period_start
            periods.append({
                'parameter_value': current_value,
                'from_date': period_start,
                'to_date': period_end,
                'duration_minutes': duration.total_seconds() / 60,
                'start_idx': len(df)-1,
                'end_idx': len(df)-1
            })
            
            # 4. Remove periods less than 45 minutes
            valid_periods = [p for p in periods if p['duration_minutes'] >= 45]
            logger.info(f"Removed {len(periods) - len(valid_periods)} periods shorter than 45 minutes")
            
            # Process each valid period
            processed_segments = []
            for period in valid_periods:
                # Get segment data
                segment_mask = (
                    (df['DateTime'] >= period['from_date']) & 
                    (df['DateTime'] <= period['to_date'])
                )
                segment = df[segment_mask].copy()
                
                # 3. Skip first 15 minutes
                cutoff_time = period['from_date'] + timedelta(minutes=15)
                valid_data = segment[segment['DateTime'] >= cutoff_time]
                
                if not valid_data.empty:
                    # Calculate averages for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    avg_values = valid_data[numeric_cols].mean()
                    avg_values['DateTime'] = period['from_date']
                    avg_values[quality_parameter] = period['parameter_value']
                    avg_values['ProcessedTimestamp'] = pd.Timestamp.now()
                    avg_values['SourceRecords'] = len(valid_data)
                    processed_segments.append(avg_values)
            
            # Convert to DataFrame
            result_df = pd.DataFrame(processed_segments)
            
            # 5. Shift averages up one step
            if len(result_df) > 1:
                # Get numeric columns to shift (excluding metadata and parameter columns)
                shift_cols = [col for col in result_df.columns 
                            if col not in ['DateTime', quality_parameter, 'ProcessedTimestamp', 'SourceRecords']]
                
                # Store first row values
                first_row = result_df[shift_cols].iloc[0].copy()
                
                # Shift values up
                result_df[shift_cols] = result_df[shift_cols].shift(-1)
                
                # Set last row to use same values as previous row
                result_df.iloc[-1][shift_cols] = result_df.iloc[-2][shift_cols]
                
                # Restore first row values
                result_df.iloc[0][shift_cols] = first_row
                
                logger.info("Shifted averages up one step relative to parameter values")
            
            # Add final processing timestamp
            result_df['ProcessedTimestamp'] = pd.Timestamp.now()
            
            logger.info(f"Successfully processed {len(df)} records into {len(result_df)} segments")
            logger.info(f"Average records per segment: {result_df['SourceRecords'].mean():.2f}")
            
            return result_df

        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

    def filter_data(self, processed_data: Dict[str, pd.DataFrame],
                filter_conditions: Dict[str, Dict[str, float]]) -> Dict[str, pd.DataFrame]:
        """
        Apply filters to processed data.
       
        Args:
            processed_data (Dict[str, pd.DataFrame]): Dictionary of processed silo data
            filter_conditions (Dict[str, Dict[str, float]]): Filtering conditions
           
        Returns:
            Dict[str, pd.DataFrame]: Filtered data
        """
        try:
            filtered_data = {}
            for silo, data in processed_data.items():
                filtered_df = data.copy()
                for column, conditions in filter_conditions.items():
                    # Handle column names with spaces
                    if column in filtered_df.columns:
                        col_name = column
                    else:
                        # Try with underscores instead of spaces
                        col_name = column.replace(' ', '_')
                        if col_name not in filtered_df.columns:
                            logger.warning(f"Column {column} not found in data for {silo}")
                            continue
                            
                    for operation, value in conditions.items():
                        if operation == 'min':
                            filtered_df = filtered_df[filtered_df[col_name] >= value]
                        elif operation == 'max':
                            filtered_df = filtered_df[filtered_df[col_name] <= value]
               
                if not filtered_df.empty:
                    filtered_data[silo] = filtered_df
                    logger.info(f"After filteration,Remaining data records: {len(filtered_df)} for {silo}")
               
            return filtered_data
           
        except Exception as e:
            logger.error(f"Error in filtering data: {str(e)}")
            raise

    def find_optimal_clusters_dendrogram(self, linkage_matrix: np.ndarray, max_clusters: int = 5) -> int:
        """
        Find optimal number of clusters by identifying the largest vertical gap in the dendrogram.
       
        Args:
            linkage_matrix: The linkage matrix from hierarchical clustering
            max_clusters: Maximum number of clusters to consider
           
        Returns:
            int: Optimal number of clusters
        """
        # Get the heights (distances) at which clusters merge
        heights = linkage_matrix[:, 2]
        heights = np.sort(heights)[::-1]  # Sort in descending order
       
        # Calculate gaps between consecutive heights
        gaps = np.diff(heights)
       
        # Find the largest gap within our cluster range constraint
        n_samples = len(linkage_matrix) + 1
        valid_gaps = gaps[:min(max_clusters-1, n_samples-2)]
        if len(valid_gaps) == 0:
            return 4  # Default to 5 clusters if we can't find valid gaps
           
        largest_gap_idx = np.argmax(valid_gaps)
        # Number of clusters is index + 2 (since we start with n_samples clusters and merge down)
        optimal_clusters = largest_gap_idx + 2
       
        return optimal_clusters

    def analyze_clusters(self, X_scaled: np.ndarray, max_clusters: int = 5) -> tuple:
        """
        Analyze optimal number of clusters using both dendrogram and silhouette score.
       
        Args:
            X_scaled: Scaled input data
            max_clusters: Maximum number of clusters to consider
           
        Returns:
            tuple: (optimal_k, linkage_matrix, silhouette_scores)
        """
        try:
            # Calculate linkage matrix for dendrogram
            linkage_matrix = shc.linkage(X_scaled, method='ward')
           
            # Calculate silhouette scores
            silhouette_scores = []
            range_n_clusters = range(2, max_clusters + 1)
           
            for n_clusters in range_n_clusters:
                clusterer = Agglomerative(n_clusters=n_clusters)
                cluster_labels = clusterer.fit_predict(X_scaled)
               
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append((n_clusters, silhouette_avg))
           
            if not silhouette_scores:
                # If we couldn't calculate silhouette scores, default to 4 clusters
                return 4, linkage_matrix, [(4, 0.0)]
           
            # Find optimal clusters using dendrogram
            dendrogram_k = self.find_optimal_clusters_dendrogram(linkage_matrix)
           
            # Get silhouette score for dendrogram-based clusters
            dendrogram_labels = Agglomerative(n_clusters=dendrogram_k).fit_predict(X_scaled)
            dendrogram_silhouette = silhouette_score(X_scaled, dendrogram_labels)
           
            # Compare dendrogram-based clusters with best silhouette score
            best_silhouette_k, best_silhouette = max(silhouette_scores, key=lambda x: x[1])
           
            # Choose optimal k based on both methods
            if dendrogram_silhouette >= 0.95 * best_silhouette:
                optimal_k = dendrogram_k  # Prefer dendrogram if scores are close
            else:
                optimal_k = best_silhouette_k
           
            return optimal_k, linkage_matrix, silhouette_scores
           
        except Exception as e:
            logger.error(f"Error in cluster analysis: {str(e)}")
            # Return default values in case of error
            return 4, linkage_matrix, [(4, 0.0)]

    def plot_cluster_analysis(self, silo: str, X_scaled: np.ndarray,
                            linkage_matrix: np.ndarray, silhouette_scores: list,
                            optimal_k: int, viz_dir: str, data: pd.DataFrame = None, 
                            features: List[str] = None) -> dict:
        """
        Create comprehensive clustering analysis plots including:
        1. Dendrogram and Silhouette Scores
        2. Feature distributions by cluster
        3. Operational pattern visualization (feature ranges by cluster)
       
        Args:
            silo: Name of the silo
            X_scaled: Scaled input data
            linkage_matrix: Linkage matrix for dendrogram
            silhouette_scores: List of silhouette scores
            optimal_k: Optimal number of clusters
            viz_dir: Directory to save visualizations
            data: Original DataFrame with features
            features: List of feature names
       
        Returns:
            dict: Paths to saved visualizations
        """
        try:
            viz_paths = {}
            
            # 1. Plot Dendrogram and Silhouette Scores
            fig = plt.figure(figsize=(15, 10))
           
            # Plot Dendrogram
            plt.subplot(2, 1, 1)
            shc.dendrogram(linkage_matrix, no_labels=True)
            plt.title(f"Clustering Analysis for {silo}")
            plt.xlabel("Samples")
            plt.ylabel("Distance")
           
            # Add cut line for optimal clusters
            if len(linkage_matrix) >= optimal_k - 1:
                cut_height = np.mean(linkage_matrix[-(optimal_k-1):-(optimal_k-2), 2]) if optimal_k > 1 else 0
                plt.axhline(y=cut_height, color='r', linestyle='--',
                        label=f'Cut line (n_clusters={optimal_k})')
                plt.legend()
           
            # Plot Silhouette Scores
            plt.subplot(2, 1, 2)
            n_clusters, scores = zip(*silhouette_scores)
            plt.plot(n_clusters, scores, marker='o', label="Silhouette Score")
            plt.axvline(optimal_k, color='r', linestyle='dotted',
                    label=f"Optimal Clusters: {optimal_k}")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Score")
            plt.legend()
           
            plt.tight_layout()
            analysis_path = os.path.join(viz_dir, f'{silo}_cluster_analysis.png')
            plt.savefig(analysis_path)
            plt.close()
            viz_paths['cluster_analysis'] = analysis_path

            # 2. Plot feature distributions for each cluster
            if data is not None and features is not None:
                # Perform clustering with optimal k
                clusterer = Agglomerative(n_clusters=optimal_k)
                labels = clusterer.fit_predict(X_scaled)

                # Create a figure for all features
                n_features = len(features)
                n_cols = 2  # Number of columns in the subplot grid
                n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division for number of rows

                fig = plt.figure(figsize=(15, 5 * n_rows))
                fig.suptitle(f"Feature Distributions by Cluster - {silo}", fontsize=16, y=1.02)

                for i, feature in enumerate(features):
                    plt.subplot(n_rows, n_cols, i + 1)
                    
                    # Plot distribution for each cluster
                    for label in range(optimal_k):
                        cluster_data = data[feature][labels == label]
                        sns.kdeplot(data=cluster_data, label=f'Cluster {label}')
                        
                        # Add cluster mean line
                        mean_val = cluster_data.mean()
                        plt.axvline(mean_val, 
                                  color=plt.gca().lines[-1].get_color(),
                                  linestyle='--', 
                                  alpha=0.5,
                                  label=f'Mean C{label}: {mean_val:.2f}')
                    
                    plt.title(f'{feature} Distribution')
                    plt.xlabel(feature)
                    plt.ylabel('Density')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.tight_layout()
                distributions_path = os.path.join(viz_dir, f'{silo}_feature_distributions.png')
                plt.savefig(distributions_path, bbox_inches='tight')
                plt.close()
                viz_paths['feature_distributions'] = distributions_path

                # 3. Create  pattern visualization (box plots)
                plt.figure(figsize=(15, 8))
                
                # Calculate cluster statistics
                cluster_stats = []
                for feature in features:
                    for label in range(optimal_k):
                        cluster_data = data[feature][labels == label]
                        stats = {
                            'Feature': feature,
                            'Cluster': f'Cluster {label}',
                            'Mean': cluster_data.mean(),
                            'Min': cluster_data.min(),
                            'Max': cluster_data.max(),
                            '25th': cluster_data.quantile(0.25),
                            '75th': cluster_data.quantile(0.75)
                        }
                        cluster_stats.append(stats)
                
                stats_df = pd.DataFrame(cluster_stats)
                
                # Create a grouped box plot
                plt.figure(figsize=(15, 8))
                positions = np.arange(len(features)) * (optimal_k + 1)
                width = 0.8
                
                for i in range(optimal_k):
                    cluster_data = stats_df[stats_df['Cluster'] == f'Cluster {i}']
                    
                    # Plot boxes
                    boxes = plt.boxplot([
                        [row['Min'], row['25th'], row['Mean'], row['75th'], row['Max']]
                        for _, row in cluster_data.iterrows()
                    ], positions=positions + i * width, widths=width,
                    patch_artist=True, labels=features,
                    medianprops=dict(color='black'),
                    boxprops=dict(facecolor=plt.cm.Set3(i / optimal_k)))
                
                plt.title(f'Patterns by Cluster - {silo}')
                plt.xlabel('Features')
                plt.ylabel('Value Range')
                plt.xticks(positions + (optimal_k - 1) * width / 2, features, rotation=45)
                
                # Add legend
                legend_elements = [
                    Patch(facecolor=plt.cm.Set3(i / optimal_k), label=f'Cluster {i}')
                    for i in range(optimal_k)
                ]
                plt.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                patterns_path = os.path.join(viz_dir, f'{silo}_patterns.png')
                plt.savefig(patterns_path, bbox_inches='tight')
                plt.close()
                viz_paths['patterns'] = patterns_path

            return viz_paths
           
        except Exception as e:
            logger.error(f"Error in plotting cluster analysis: {str(e)}")
            return viz_paths

    def perform_clustering(self, filtered_data: Dict[str, pd.DataFrame],
                     features: List[str]) -> Dict[str, Any]:
        try:
            results = {
                'summary': {},
                'cluster_stats': {},
                'visualizations': {}
            }
           
            for silo, data in filtered_data.items():
                # Handle feature names with spaces
                available_features = []
                for feature in features:
                    if feature in data.columns:
                        available_features.append(feature)
                    else:
                        # Try with underscores instead of spaces
                        feature_underscore = feature.replace(' ', '_')
                        if feature_underscore in data.columns:
                            available_features.append(feature_underscore)
                        else:
                            logger.warning(f"Feature {feature} not found in data for {silo}")
                            continue

                if not data.empty and len(available_features) == len(features):
                    # Initialize visualizations dictionary for this silo
                    results['visualizations'][silo] = {
                        'feature_distributions': {}
                    }
                   
                    # Prepare data for clustering
                    X = data[available_features].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                   
                    # Create visualizations directory
                    viz_dir = os.path.join(os.getcwd(), 'visualizations', silo)
                    os.makedirs(viz_dir, exist_ok=True)
                   
                    # Perform cluster analysis
                    max_clusters = min(5, len(X_scaled) - 1)
                    optimal_k, linkage_matrix, silhouette_scores = self.analyze_clusters(
                        X_scaled, max_clusters)
                   
                    # Create visualization plots
                    try:
                        viz_paths = self.plot_cluster_analysis(
                            silo, X_scaled, linkage_matrix, silhouette_scores, optimal_k, viz_dir,
                            data=data, features=available_features)
                        # Update visualizations dictionary with new paths
                        results['visualizations'][silo].update(viz_paths)
                    except Exception as e:
                        logger.error(f"Error creating cluster analysis plots for {silo}: {str(e)}")
                   
                    # Perform final clustering with optimal number of clusters
                    clustering = Agglomerative(n_clusters=optimal_k)
                    labels = clustering.fit_predict(X_scaled)
                   
                    # Calculate final silhouette score
                    final_silhouette_score = silhouette_score(X_scaled, labels)
                   
                    # Calculate cluster statistics
                    unique_labels = np.unique(labels)
                    cluster_sizes = {str(label): np.sum(labels == label) for label in unique_labels}
                   
                    results['summary'][silo] = {
                        'n_clusters': len(unique_labels),
                        'n_samples': len(labels),
                        'cluster_sizes': cluster_sizes,
                        'optimal_k': optimal_k,
                        'silhouette_score': final_silhouette_score
                    }
                   
                    # Calculate detailed cluster statistics
                    results['cluster_stats'][silo] = {}
                    for label in unique_labels:
                        cluster_mask = labels == label
                        cluster_data = data[available_features][cluster_mask]
                       
                        results['cluster_stats'][silo][str(label)] = {
                            'size': np.sum(cluster_mask),
                            'mean': cluster_data.mean().to_dict(),
                            'std': cluster_data.std().to_dict(),
                            'min': cluster_data.min().to_dict(),
                            '25th_percentile': cluster_data.quantile(0.25).to_dict(),
                            'median': cluster_data.median().to_dict(),
                            '75th_percentile': cluster_data.quantile(0.75).to_dict(),
                            'max': cluster_data.max().to_dict()
                        }
                   
                    logger.info(f"Completed clustering for {silo} with {optimal_k} clusters (Silhouette Score: {final_silhouette_score:.3f})")
           
            return results
           
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            raise

    def run_clustering_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main function that executes the entire clustering pipeline.
       
        Args:
            config (Dict[str, Any]): Configuration dictionary
           
        Returns:
            Dict[str, Any]: Final results of the clustering pipeline
        """
        # Validate configuration
        required_keys = ['data_path', 'quality_parameter', 'features', 'filter_conditions']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required configuration key: '{key}'")
 
        try:
            logger.info("Starting Clustering Pipeline...")
           
            # Load data
            df = pd.read_excel(config['data_path'])
            logger.info(f"Loaded {len(df)} records")
 
            # 1. First segregate by SILO
            silo_data = self.segregate_silo_data(df)
            logger.info(f"Segregated into {len(silo_data)} silos")
 
            # 2. Process SILO data
            processed_data = self.process_silo_data(silo_data)
            if not processed_data:
                raise ValueError("No data remained after processing")
            logger.info(f"Processed data for {len(processed_data)} silos")

            # 3. Convert to hourly data for each processed silo
            hourly_data = {}
            for silo, data in processed_data.items():
                hourly_data[silo] = self.extract_and_convert_hourly(data, config['quality_parameter'])
                logger.info(f"Converted {silo} to {len(hourly_data[silo])} hourly records")
 
            # Apply filters
            filtered_data = self.filter_data(hourly_data, config['filter_conditions'])
            if not filtered_data or all(df.empty for df in filtered_data.values()):
                logger.warning("No valid data available for clustering")
                return {}
 
            # Perform clustering
            clustering_results = self.perform_clustering(filtered_data, config['features'])
            if not clustering_results:
                raise ValueError("No clustering results generated")
 
            # Compile results
            final_results = {
                'config': config,
                'raw_records': len(df),
                'silo_count': len(silo_data),
                'processed_silo_count': len(processed_data),
                'hourly_records': sum(len(df) for df in hourly_data.values()),
                'filtered_silo_count': len(filtered_data),
                'processed_records': sum(len(df) for df in processed_data.values()),
                'filtered_records': sum(len(df) for df in filtered_data.values()),
                'clustering_results': clustering_results
            }
 
            logger.info("Pipeline completed successfully!")
            return final_results
 
        except Exception as e:
            logger.error(f"Error in clustering pipeline: {str(e)}")
            raise