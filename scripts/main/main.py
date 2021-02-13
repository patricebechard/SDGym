
import pickle as pkl
import logging
import sdgym
from sdgym.data import load_dataset
from sdgym.synthesizers import (IdentitySynthesizer, CLBNSynthesizer, CTGANSynthesizer)

logging.basicConfig(level=logging.INFO)

def load_synthesizer(synthesizer_name, dataset_name):
    pass

def save_synthesizer(synthesizer, synthesizer_name, dataset_name):
    synthesizer_filename = "%s_%s.pkl" % (synthesizer_name, dataset_name)

    if synthesizer_name == "IdentitySynthesizer":
        with open(synthesizer_filename, "wb") as f:
            pkl.dump(synthesizer, f)


def main(dataset_name, synthesizer):
    synthesizer_name = synthesizer.__class__.__name__
    
    # load dataset
    logging.info("Loading %s dataset" % dataset_name)
    train, test, meta, categoricals, ordinals = load_dataset(dataset_name, benchmark=True)

    # fit synthesizer
    logging.info("Fitting %s synthesizer on %s dataset" % (synthesizer_name, dataset_name))
    synthesizer.fit(train, categoricals, ordinals)

    # save synthesizer
    logging.info("Saving %s synthesizer trained on %s dataset" % (synthesizer_name, dataset_name))
    save_synthesizer(synthesizer, synthesizer_name, dataset_name)
    

if __name__ == "__main__":

    all_synthesizers = [
        IdentitySynthesizer,
        CLBNSynthesizer,
        CTGANSynthesizer,
    ]

    all_datasets = [
        "california-housing",
        "titanic",
        "porto-seguro",
    ]

    for synthesizer_class in all_synthesizers:
        for dataset_name in all_datasets:

            synthesizer = synthesizer_class()
            main(dataset_name=dataset_name, synthesizer=synthesizer)
