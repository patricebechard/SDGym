import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

RAW_DATASET_PATH = os.path.realpath("../../sdgym/data/porto-seguro/train.csv.zip")
NP_DATASET_PATH = os.path.realpath("../../sdgym/data/porto-seguro.npz")
JSON_FILE_PATH = os.path.realpath("../../sdgym/data/porto-seguro.json")

def create_ordinal_column(data_df, col_name, label_col_name="target"):

    i2s = [str(val) for val in sorted(data_df[col_name].unique())]
    col_size = len(i2s)
    col_type = "ordinal"
    if col_name == label_col_name:
        col_name = "label"

    column_object = {
        "i2s": i2s,
        "name": col_name,
        "size": col_size,
        "type": col_type
    }
    return column_object

def create_categorical_column(data_df, col_name, label_col_name="target"):

    i2s = [str(val) for val in sorted(data_df[col_name].unique())]
    col_size = len(i2s)
    col_type = "categorical"
    if col_name == label_col_name:
        col_name = "label"

    column_object = {
        "i2s": i2s,
        "name": col_name,
        "size": col_size,
        "type": col_type
    }
    return column_object

def create_continuous_column(data_df, col_name):
    values = data_df[col_name]
    min_value = min(values)
    max_value = max(values)
    col_type = "continuous"

    column_object = {
        "max": max_value,
        "min": min_value,
        "name": col_name,
        "type": col_type
    }
    return column_object

def create_json_file(data_df):

    json_object = {}

    # add columns to json object
    json_object["columns"] = []
    for col_name in data_df.columns:

        if col_name == "target" or col_name.endswith("_cat") or col_name.endswith("_bin"):
            column_object = create_categorical_column(data_df, col_name)
        elif type(data_df[col_name].values[0]) == np.float64:
            column_object = create_continuous_column(data_df, col_name)
        elif type(data_df[col_name].values[0]) == np.int64:
            column_object = create_ordinal_column(data_df, col_name)

        json_object["columns"].append(column_object)
    json_object["problem_type"] = "binary_classification"

    with open(JSON_FILE_PATH, "w") as f:
        json.dump(json_object, f, indent=4)

def create_npz_file(data_df):

    for col_name in data_df.columns:

        # categorical column
        if col_name == "target" or col_name.endswith("_cat") or col_name.endswith("_bin"):
            vals = sorted(data_df[col_name].unique())
            mapping = {str(val): i for i, val in enumerate(vals)}
            data_df[col_name] = data_df[col_name].apply(lambda x: mapping[str(x)])

        # continuous column
        elif type(data_df[col_name].values[0]) == np.float64:
            pass

        # ordinal column
        elif type(data_df[col_name].values[0]) == np.int64:
            vals = sorted(data_df[col_name].unique())
            mapping = {str(val): i for i, val in enumerate(vals)}
            data_df[col_name] = data_df[col_name].apply(lambda x: mapping[str(x)])

    train_data_df, test_data_df = train_test_split(data_df, test_size=0.25)

    train_data_np = train_data_df.values
    test_data_np = test_data_df.values
    np.savez_compressed(NP_DATASET_PATH, train=train_data_np, test=test_data_np)

def main():

    data_df = pd.read_csv(RAW_DATASET_PATH, index_col=0)

    # create json file
    create_json_file(data_df)

    # process data df columns
    create_npz_file(data_df)

def try_out():

    import sdgym
    from sdgym.synthesizers import (CLBNSynthesizer, CTGANSynthesizer, IdentitySynthesizer)

    all_synthesizers = [
        CLBNSynthesizer,
        IdentitySynthesizer,
        CTGANSynthesizer,
    ]
    
    scores = sdgym.run(synthesizers=all_synthesizers, datasets=["porto-seguro"])
    print(scores)    


if __name__ == "__main__":
    main()
    try_out()
