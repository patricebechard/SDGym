from sdgym.data import load_dataset
import pickle as pkl
import logging
import pandas as pd

from sdv import SDV, Metadata

from sdmetrics import evaluate
from synthetic_data.evaluation.latent_structure import LogCluster
from synthetic_data.evaluation.correlation import SpearmanCorrelation, TheilsU, KruskalWillisHTest

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

    meta = Metadata()
    meta.add_table("the_table", data)

    synth_data = synthesizer.sample(len(train))
    synth_data = pd.DataFrame(synth_data, columns=[elem['name'] for elem in metadata['columns']])

    report = evaluate(meta, {"the_table": data}, {"the_table": synth_data})
    report.add_metrics(LogCluster().metrics(meta, {"the_table": data}, {"the_table": synth_data}))
    report.add_metrics(SpearmanCorrelation().metrics(meta, {"the_table": data}, {"the_table": synth_data}))
    report.add_metrics(TheilsU().metrics(meta, {"the_table": data}, {"the_table": synth_data}))
    report.add_metrics(KruskalWillisHTest().metrics(meta, {"the_table": data}, {"the_table": synth_data}))

    print(report.details().groupby("Name").mean())

if __name__ == "__main__":
    # synthesizer_name = "IdentitySynthesizer"
    # synthesizer_name = "CTGANSynthesizer"
    synthesizer_name = "GaussianCopulaCategorical"

    dataset_name = "california-housing"
    # dataset_name = "titanic"
    # dataset_name = "porto-seguro"

    main(synthesizer_name, dataset_name)