#! /usr/bin/python
import numpy as np

def manhattan_distance(pt1, pt2):
	return np.sum([np.abs(p1 - p2) for p1, p2 in zip(pt1, pt2)])
	
def euclidean_distance(pt1, pt2):
	return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

group_red = [(4.7, 3.2), (4.9, 3.1), (5.0, 3.0), (4.6, 2.9)]        # n
group_blue = [(5.9, 3.2), (6.7, 3.1), (6.0, 3.0), (6.2, 2.8)]       # m


min_distance = float("inf")
max_distance = 0

distance = euclidean_distance
points_max = None
points_min = None
distances = []

# O(n*m)
for red_pt in group_red:
    for blue_pt in group_blue:
        dist = distance(red_pt, blue_pt)
        distances.append(dist)
        if dist > max_distance:
            max_distance = dist
            points_max = [red_pt, blue_pt]
        if dist < min_distance:
            min_distance = dist
            points_min = [red_pt, blue_pt]

print(f"min: {min_distance:.2f} {points_min}")
print(f"max: {max_distance:.2f} {points_max}")
avg = sum(distances) / len(distances)
print(f"avg: {avg:.2f}")

min_distance = float("inf")
max_distance = 0
points_max = None
points_min = None
distances = []
for pt_a in group_red:
    for pt_b in group_red:
        if pt_a == pt_b: continue
        dist = distance(pt_a, pt_b)
        distances.append(dist)
        if dist > max_distance:
            max_distance = dist
            points_max = [pt_a, pt_b]
        if dist < min_distance:
            min_distance = dist
            points_min = [pt_a, pt_b]

print('red group')
print(f"min: {min_distance:.2f} {points_min}")
print(f"max: {max_distance:.2f} {points_max}")
avg = sum(distances) / len(distances)
print(f"avg: {avg:.2f}")

min_distance = float("inf")
max_distance = 0
points_max = None
points_min = None
distances = []
for pt_a in group_blue:
    for pt_b in group_blue:
        if pt_a == pt_b: continue
        dist = distance(pt_a, pt_b)
        distances.append(dist)
        if dist > max_distance:
            max_distance = dist
            points_max = [pt_a, pt_b]
        if dist < min_distance:
            min_distance = dist
            points_min = [pt_a, pt_b]

print('blue group')
print(f"min: {min_distance:.2f} {points_min}")
print(f"max: {max_distance:.2f} {points_max}")
avg = sum(distances) / len(distances)
print(f"avg: {avg:.2f}")
