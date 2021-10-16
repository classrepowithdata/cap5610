from argparse import ArgumentParser
from random import randint
from sys import exit

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def euclidean_distance(data, centroid):
    return np.sum((data - centroid)**2, axis=1)

def manhattan_distance(data, centroid):
    return np.sum(np.abs(data - centroid), axis=1)
 
def parse_centroid(center):
    return [float(f) for f in center.split(",")]

def main(args):
    data = pd.read_csv(args.data_file)    
    data = data.loc[:, args.data_column].values

    training_samples = data.shape[0]
    features = data.shape[1]
    
    centroids = np.zeros((args.total_centroids, features))
    
    if args.centroid:
        # centroids are given to us
        for count, centroid in enumerate(args.centroid):
            centroids[count] = parse_centroid(centroid)
    else:
        # get some random centroids from the dataset
        centroids_used = []
        for i in range(args.total_centroids):
            while (rand := randint(0, training_samples - 1)) in centroids_used:
                pass
            centroids_used.append(rand)
            centroids[i] = data[rand]

    distance_formula = {
            "euclidean":    euclidean_distance,
            "manhattan":    manhattan_distance
        }[args.distance_type]
    
    print(data)
    print(centroids)

    for iteration in range(args.iterations):
        distances = np.array([]).reshape(training_samples, 0)
        for k in range(args.total_centroids):
            dist = distance_formula(data, centroids[k])
            distances = np.c_[distances, dist]
        min_distances = np.argmin(distances, axis=1)
        print(distances)
        print(min_distances)
        
        # prepare a dictionary to store the groups of points
        tmp_dict = {}
        for centroid in range(args.total_centroids):
            tmp_dict[centroid] = [] #np.array([]).reshape(2, 0)
        
        # store points in groups based off the closeness to a centroid
        for i in range(training_samples):
            min_distance = min_distances[i]
            tmp_dict[min_distance].append(data[i]) # = np.c_[tmp_dict[min_distance], data[i]]
        for centroid in range(args.total_centroids):
            tmp_dict[centroid] = np.array(tmp_dict[centroid])#.T
        
        # update position for centroids
        for centroid in range(args.total_centroids):
            centroids[centroid] = np.mean(tmp_dict[centroid], axis=0)

        print(f"{iteration+1: 3d}: centroids\n{centroids}")

    # plot results
    colors = ['blue', 'red', 'green', 'brown', 'cyan', 'magenta']
    for i in range(args.total_centroids):
        plt.scatter(tmp_dict[i][:,0], tmp_dict[i][:,1], c=colors[i % len(colors)], label=f"cluster {i+1}")
    plt.scatter(centroids[:,0], centroids[:,1], c='yellow', s=250, label="centroid")
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--distance-type", type=str, action="store", \
        default="euclidean", choices=["euclidean", "manhattan"],
        help="distance formula to use")
    parser.add_argument("--data-file", type=str, action="store", \
        default="data.csv")
    parser.add_argument("--centroid", action="append", default=None)
    parser.add_argument("--total-centroids", action="store", type=int, default=2)
    parser.add_argument("--data-column", action="append", type=str, default=["x1", "x2"])
    parser.add_argument("--iterations", action="store", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.centroid is not None and len(args.centroid) == 1:
        parser.error("must specify more than one --centroid if using this option")
        exit(1)
    elif args.centroid is not None:
        args.total_centroids = len(args.centroid)
    
    exit(main(args))
    