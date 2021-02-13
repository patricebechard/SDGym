# Porto Seguro's Safe Driver Prediction Dataset

From Kaggle :

> Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.

> Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

> In this competition, you’re challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.

Link to Kaggle competition : https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/

## Setup

To setup everything, download the data files from Kaggle. You should download 3 distinct files : `train.csv.zip`, `test.csv.zip`, and `sample_submission.csv`. Only the train file will be used.

Make sure to put these files in this location in a newly created directory: `sdgym/data/porto-seguro`

## Usage

Once you have downloaded the data, you can create the `.json` and `.npz` files needed by SDGym by running this script from this specific location (here):

    python preprocess_porto_seguro.py


This will do four things:
* Create the `porto-seguro.json` file needed by SDGym
* Preprocess data, split in train/val, and save as `porto-seguro.npz` file needed by SDGym
* Run and evaluate on data synthesizers on a binary classificationt task.
