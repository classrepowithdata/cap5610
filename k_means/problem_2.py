from argparse import ArgumentParser
from random import randint
from sys import exit

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd


def euclidean_distance(data, centroid):
    return np.sum((data - centroid)**2, axis=1)

def manhattan_distance(data, centroid):
    return np.sum(np.abs(data - centroid), axis=1)
    
def jaccard_distance(data, centroid):
    ret = np.zeros((data.shape[0], 1))
    for count, point in enumerate(data):
        mask = point < centroid
        numerator = np.sum(point[mask]) + np.sum(centroid[~mask])
        denominator = np.sum(point[~mask]) + np.sum(centroid[mask])
        ret[count] = 1 - numerator/denominator
        
    return ret
    
def cosine_similarity(data, centroid):
    ret = np.zeros((data.shape[0], 1))
    for count, point in enumerate(data):
        dist = np.dot(centroid, point) / (LA.norm(centroid) * LA.norm(point))
        ret[count] = dist
    #print(ret)
    #exit(1)
    return ret

def positions_changed(dict_a, dict_b):
    if len(dict_a.keys()) != len(dict_b.keys()):
        return True
    
    for k, points_b in dict_b.items():
        points_a = dict_a[k]
        if not np.array_equal(points_a, points_b):
            return True
    return False

def compute_sse(clusters, centroids, distance):
    sse = 0
    for index, items in clusters.items():
        sse += np.sum(distance(items, centroids[index]))
    
    return sse
    
def points_are_equal(pt1, pt2):
    if pt2.shape != pt1.shape: return False
    
    for a, b in zip(pt1, pt2):
        if a != b: return False
        
    return True
 
def parse_centroid(center):
    return [float(f) for f in center.split(",")]

def main(args):
    data = pd.read_csv(args.data_file)    
    data = data.values

    training_samples = data.shape[0]
    features = data.shape[1]
    
    centroids = np.zeros((args.total_centroids, features))
    classifications = np.zeros(training_samples)
    
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
            "manhattan":    manhattan_distance,
            "jaccard":      jaccard_distance,
            "cosine":       cosine_similarity
        }[args.distance_type]
    
    #print(data)
    #print(centroids)
    
    iteration = 0

    old_dict = {}
    old_sse = float("inf")

    while True:
        distances = np.array([]).reshape(training_samples, 0)
        for k in range(args.total_centroids):
            dist = distance_formula(data, centroids[k])
            distances = np.c_[distances, dist]
        min_distances = np.argmin(distances, axis=1)
        #print(distances)
        #print(min_distances)
        
        # prepare a dictionary to store the groups of points
        tmp_dict = {}
        for centroid in range(args.total_centroids):
            tmp_dict[centroid] = [] #np.array([]).reshape(2, 0)
        
        # store points in groups based off the closeness to a centroid
        for i in range(training_samples):
            min_distance = min_distances[i]
            tmp_dict[min_distance].append(data[i]) # = np.c_[tmp_dict[min_distance], data[i]]
            classifications[i] = min_distance
        for centroid in range(args.total_centroids):
            tmp_dict[centroid] = np.array(tmp_dict[centroid])#.T
        
        # update position for centroids
        for centroid in range(args.total_centroids):
            centroids[centroid] = np.mean(tmp_dict[centroid], axis=0)
            
        iteration += 1
                
        if not positions_changed(old_dict, tmp_dict):
            print(f"[I] positions did not change after {iteration} iterations, bailing out")
            break
        if iteration == args.iterations:
            print("[I] iteration maximum reached, bailing out")
            break
        
        new_sse = compute_sse(tmp_dict, centroids, distance_formula)
        if new_sse > old_sse and args.sse_increase:
            print(f"[I] SSE increase at iteration {iteration}, bailing out")
            break
        
        old_dict = tmp_dict.copy()
        old_sse = new_sse

    print(f"SSE: {new_sse}")

    ground_truth = pd.read_csv(args.data_labels).values
    group_data = np.zeros((args.total_centroids, args.total_centroids))
    
    # majority votes in O(n^2) because we take a shot in complexity regardless of what we do
    # and np.where() refuses to work for searching for the points...
    for predicted, points in tmp_dict.items():
        for point in points:
            # locate the index for that point
            for index, data_point in enumerate(data):
                if points_are_equal(data_point, point):
                    break
            # print(index)
            truth = ground_truth[index][0]
            group_data[predicted][truth] += 1

    maximums = np.argmax(group_data, axis=1)    
    hits = np.sum(group_data[maximums])
    print(f"[I] accuracy {hits}/{training_samples} ({100* hits/training_samples}%)")

    return 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--distance-type", type=str, action="store", \
        default="euclidean", choices=["euclidean", "manhattan", "jaccard", "cosine"],
        help="distance formula to use")
    parser.add_argument("--data-file", type=str, action="store", \
        default="data.csv")
    parser.add_argument("--data-labels", type=str, action="store", \
        default="labels.csv")
    parser.add_argument("--centroid", action="append", default=None)
    parser.add_argument("--total-centroids", action="store", type=int, default=10)
    parser.add_argument("--sse-increase", action="store_true", default=False)
    parser.add_argument("--iterations", action="store", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.centroid is not None and len(args.centroid) == 1:
        parser.error("must specify more than one --centroid if using this option")
        exit(1)
    elif args.centroid is not None:
        args.total_centroids = len(args.centroid)
    
    exit(main(args))
    