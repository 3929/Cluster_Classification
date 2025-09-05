import pytest
import pandas as pd
import numpy as np
from FINAL import CLUSTER

@pytest.fixture
def cluster():
    return CLUSTER()

def test_segregate_silo_data_basic(cluster):
    df = pd.DataFrame({
        'Timestamp': ['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00'],
        'Silo1Sel': [1, 0, 0],
        'Silo2Sel': [0, 1, 0],
        'Silo3Sel': [0, 0, 1],
        'Value': [10, 20, 30]
    })
    result = cluster.segregate_silo_data(df)
    assert set(result.keys()) == {'Silo1', 'Silo2', 'Silo3'}
    assert all('DateTime' in silo_df.columns for silo_df in result.values())
    assert result['Silo1'].iloc[0]['Value'] == 10
    assert result['Silo2'].iloc[0]['Value'] == 20
    assert result['Silo3'].iloc[0]['Value'] == 30

def test_segregate_silo_data_datetime_rename(cluster):
    df = pd.DataFrame({
        'MyDate': ['2024-01-01', '2024-01-02'],
        'Silo1Sel': [1, 0],
        'Silo2Sel': [0, 1]
    })
    result = cluster.segregate_silo_data(df)
    for silo_df in result.values():
        assert 'DateTime' in silo_df.columns

def test_segregate_silo_data_no_datetime(cluster):
    df = pd.DataFrame({
        'NotADate': [1, 2],
        'Silo1Sel': [1, 0],
        'Silo2Sel': [0, 1]
    })
    with pytest.raises(ValueError, match="No valid datetime column found in DataFrame"):
        cluster.segregate_silo_data(df)

def test_segregate_silo_data_no_silo_column(cluster):
    df = pd.DataFrame({
        'Timestamp': ['2024-01-01', '2024-01-02'],
        'Other': [1, 2]
    })
    with pytest.raises(ValueError, match="No silo selection columns found in the data"):
        cluster.segregate_silo_data(df)

def test_segregate_silo_data_exclude_unknown(cluster):
    df = pd.DataFrame({
        'Timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Silo1Sel': [1, 0, 0],
        'Silo2Sel': [0, 0, 0]
    })
    result = cluster.segregate_silo_data(df)
    assert 'Silo1' in result
    assert all(result['Silo1']['SILO'] == 'Silo1')
    total_rows = sum(len(silo_df) for silo_df in result.values())
    assert total_rows == 1

def test_process_silo_data_filters_large_gaps(cluster):
    df = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=12, freq='h'),
        'SILO': ['Silo1'] * 12,
        'Silo1Sel': [1] * 12,
        'Value': range(12)
    })
    silo_data = {'Silo1': df}
    processed = cluster.process_silo_data(silo_data)
    assert 'Silo1' in processed
    assert len(processed['Silo1']) == 12

def test_process_silo_data_insufficient_records(cluster):
    df = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=5, freq='h'),
        'SILO': ['Silo1'] * 5,
        'Silo1Sel': [1] * 5,
        'Value': range(5)
    })
    silo_data = {'Silo1': df}
    processed = cluster.process_silo_data(silo_data)
    assert processed == {}

def test_extract_and_convert_hourly_invalid_param(cluster):
    df = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=5, freq='h'),
        'Value': range(5)
    })
    with pytest.raises(ValueError):
        cluster.extract_and_convert_hourly(df, 'NotAColumn')

def test_filter_data_min_max(cluster):
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40]
    })
    processed = {'Silo1': df}
    filters = {'A': {'min': 2, 'max': 3}}
    filtered = cluster.filter_data(processed, filters)
    assert 'Silo1' in filtered
    assert all((filtered['Silo1']['A'] >= 2) & (filtered['Silo1']['A'] <= 3))

def test_find_optimal_clusters_dendrogram(cluster):
    # Create a fake linkage matrix
    linkage_matrix = np.array([
        [0, 1, 0.1, 2],
        [2, 3, 0.2, 2],
        [4, 5, 0.5, 4],
        [6, 7, 1.0, 6]
    ])
    k = cluster.find_optimal_clusters_dendrogram(linkage_matrix, max_clusters=4)
    assert isinstance(k, (int, np.integer))

def test_analyze_clusters_returns_tuple(cluster):
    X = np.random.rand(10, 2)
    optimal_k, linkage_matrix, silhouette_scores = cluster.analyze_clusters(X, max_clusters=3)
    assert isinstance(optimal_k, (int, np.integer))
    assert isinstance(linkage_matrix, np.ndarray)

