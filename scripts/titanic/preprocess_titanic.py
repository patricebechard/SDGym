import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

RAW_DATASET_PATH = os.path.realpath("../sdgym/data/titanic/train.csv")
NP_DATASET_PATH = os.path.realpath("../sdgym/data/titanic.npz")

sex_s2i = {'male': 0, "female": 1}
embarked_s2i = {"S": 0, "C": 1, "Q": 2}


def main():

    data_df = pd.read_csv(RAW_DATASET_PATH, index_col=0).drop(["Name", "Ticket", "Cabin"], axis=1)

    # infer missing values for age using mean
    data_df.Age.fillna((data_df.Age.mean()), inplace=True)

    # infer missing values for SibSp with most common (0)
    data_df.SibSp.fillna(0, inplace=True)

    # infer missing valies for Embarked with most common (S)
    data_df.Embarked.fillna("S", inplace=True)

    # translate Sex feature to int
    data_df.Sex = data_df.Sex.apply(lambda x: sex_s2i[x])

    # translate Embarled feature to int
    data_df.Embarked = data_df.Embarked.apply(lambda x: embarked_s2i[x])
    data_df.Pclass -= 1

    data_df = data_df.drop_duplicates()

    train_data_df, test_data_df = train_test_split(data_df, test_size=0.25)

    train_data_np = train_data_df.values
    test_data_np = test_data_df.values
    np.savez_compressed(NP_DATASET_PATH, train=train_data_np, test=test_data_np)


def try_out():

    import sdgym
    from sdgym.synthesizers import (
        CLBNSynthesizer, CTGANSynthesizer, IdentitySynthesizer, IndependentSynthesizer,
        MedganSynthesizer, PrivBNSynthesizer, TableganSynthesizer, TVAESynthesizer,
        UniformSynthesizer, VEEGANSynthesizer)

    all_synthesizers = [
        CLBNSynthesizer,
        IdentitySynthesizer,
        IndependentSynthesizer,
        MedganSynthesizer,
        PrivBNSynthesizer,
        TableganSynthesizer,
        CTGANSynthesizer,
        TVAESynthesizer,
        UniformSynthesizer,
        VEEGANSynthesizer,
    ]
    
    scores = sdgym.run(synthesizers=all_synthesizers, datasets=["titanic"])
    print(scores)


if __name__ == "__main__":
    main()
    try_out()
