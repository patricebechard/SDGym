# Titanic Dataset

From Kaggle :

> This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works.
> The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

Link to Kaggle page : https://www.kaggle.com/c/titanic/

To set up everything, download the data and put it in a new directory : `sdgym/data/titanic`. The files you download should be called `train.csv`, `test.csv`, and `gender_submission.csv`. Only the train file will be used by the script.

The scripts then executes the following : 

* it preprocesses some columns to be in line with the format expected by SDGym
* it saves the files as `.npz` files.
* it runs the benchmark on a variety of synthesizers on a binary classification task

Since SDGym also needs a `.json` file for the dataset to work (containing the metadata), and that this file is not easily created from the raw data, I have included it in the current directory. Just copy the file to `sdgym/data` for everything to work properly.

