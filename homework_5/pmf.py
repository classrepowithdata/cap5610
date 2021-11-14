#! /usr/bin/python
from surprise import Dataset
from surprise import Reader
from surprise import SVD, KNNBasic
from surprise.model_selection import cross_validate
from argparse import ArgumentParser
import numpy as np


def main(args):
    if args.use_builtin:
        data = Dataset.load_builtin('ml-100k')
    else:
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        data = Dataset.load_from_file(args.ratings_file, reader=reader)

    if args.model == "pmf":
        algo = SVD(biased=False)
    else:
        algo = KNNBasic(k=args.max_neighbors, sim_options = {"user_based": args.model == "ubcf", 'name': args.similarity})

    res = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    # print(res)
    print(f"{args.max_neighbors} {np.mean(res['test_rmse'])} {np.mean(res['test_mae'])}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use-builtin", action="store_true", default=False)
    parser.add_argument("--ratings-file", action="store", type=str, default='ratings.csv')
    parser.add_argument("--model", choices=["pmf", "ubcf", "ibcf"], type=str, action='store', default='pmf')
    parser.add_argument("--similarity", choices=["MSD", "cosine", "pearson"], type=str, action='store', default='MSD')
    parser.add_argument("--max-neighbors", type=int, default=40, action="store")

    args = parser.parse_args()
    main(args)