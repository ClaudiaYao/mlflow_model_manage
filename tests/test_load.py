
import sys
from unittest.mock import MagicMock
from src.utils import spark_utils

# Prevent import errors if databricks.connect is not installed
sys.modules['databricks.connect'] = MagicMock()

# Import the module after mocking databricks.connect
from src.data import load

# Test load_data with mocked config and spark.table
def test_load_data_with_mock():
        df = load.load_data(spark_utils.spark)
        assert df.count() > 0


# Test split_raw_data using a small real DataFrame
def test_split_raw_data_returns_splits():
    
    # spark = SparkSession.builder.appName("DataFrameCreation").getOrCreate()
    
    data = [(i, chr(96 + (i % 5 + 1))) for i in range(1, 31)]
    columns = ["id", "value"]
    df = spark_utils.spark.createDataFrame(data, columns)

    train_df, val_df, test_df = load.split_raw_data(df, seed=123)

    # Check that all splits are DataFrames
    assert hasattr(train_df, "count")
    assert hasattr(train_df, "schema")

    # Total rows should match original
    total_rows = train_df.count() + val_df.count() + test_df.count()
    assert total_rows == df.count()

    # Approximate split ratios
    train_ratio = train_df.count() / df.count()
    val_ratio = val_df.count() / df.count()
    test_ratio = test_df.count() / df.count()
    assert 0.6 < train_ratio < 0.8
    assert 0 < val_ratio < 0.2
    assert 0 < test_ratio < 0.2