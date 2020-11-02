from sdgym.data import load_dataset
import pickle as pkl
import logging
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score

logging.basicConfig(level=logging.INFO)

def load_synthesizer(synthesizer_name, dataset_name):
    with open(f"{synthesizer_name}_{dataset_name}.pkl", "rb") as f:
        synthesizer = pkl.load(f)
    return synthesizer

def main(synthesizer_name, dataset_name, num_samples=10000):

    # load dataset
    logging.info("Loading %s dataset" % dataset_name)
    train, test, metadata, categoricals, ordinals = load_dataset(dataset_name, benchmark=True)

    data = pd.DataFrame(train, columns=[elem['name'] for elem in metadata['columns']])

    # load synthesizer
    synthesizer = load_synthesizer(synthesizer_name, dataset_name)

    synth_data = synthesizer.sample(len(train))
    synth_data = pd.DataFrame(synth_data, columns=[elem['name'] for elem in metadata['columns']])
    synth_data_inputs = synth_data.drop(columns=["label"])
    synth_data_targets = synth_data["label"]

    test_df = pd.DataFrame(test, columns=[elem['name'] for elem in metadata['columns']])
    test_inputs = test_df.drop(columns=["label"])
    test_targets = test_df["label"]

    if dataset_name == "california-housing":
        prediction_model = XGBRegressor()
    else:
        prediction_model = XGBClassifier()

    prediction_model.fit(synth_data_inputs, synth_data_targets)

    predictions = prediction_model.predict(test_inputs)

    if dataset_name == "california-housing":
        r2 = r2_score(test_targets, predictions)
        print(f"R2 Score : {r2}")
    else:
        acc = accuracy_score(test_targets, predictions)
        f1 = f1_score(test_targets, predictions)

        print(f"Accuracy : {acc}")
        print(f"F1 Score : {f1}")




if __name__ == "__main__":
    # synthesizer_name = "IdentitySynthesizer"
    synthesizer_name = "CTGANSynthesizer"
    # synthesizer_name = "GaussianCopulaCategorical"

    # dataset_name = "california-housing"
    dataset_name = "titanic"
    # dataset_name = "porto-seguro"

    main(synthesizer_name, dataset_name)