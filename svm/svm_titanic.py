#! /usr/bin/python
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score


def main(args):
    # load data
    train_set = pd.read_csv(args.csv_file)

    # preprocess
    train_set = train_set.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
    train_set = train_set.dropna()

    target = train_set.Survived
    source = train_set[['Sex', 'Pclass', 'Age']]

    # reevaluate some things
    source.loc[source.Sex=='female', 'Sex']=1
    source.loc[source.Sex=='male', 'Sex']=0
    source["Sex"] = source["Sex"].astype(str).astype(float)

    clf = svm.SVC(kernel=args.kernel, C=1, degree=2)
    scores = cross_val_score(clf, source, target, cv=5)
    print(scores)
    
#arguments with kernel and csv file
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kernel", type=str, action="store", default="linear", choices=['linear', 'poly', 'rbf'])
    parser.add_argument("--csv-file", type=str, action="store", default="train.csv")
    args = parser.parse_args()
    main(args)
  