# import csv
# import math
#
#
# def read_distance_matrix_from_csv(csv_file):
#     distance_matrix = {}
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         header = next(reader)  # Skip the header
#         for row in reader:
#             bs_id = int(row[0])
#             distances = [float(val) for val in row[1:]]
#             distance_matrix[bs_id] = distances
#     return distance_matrix
#
# def map_actual_to_ordinal_ids(distance_matrix):
#     actual_to_ordinal_mapping = {}
#     ordinal_id = 1
#     for actual_id in sorted(distance_matrix.keys()):
#         actual_to_ordinal_mapping[actual_id] = ordinal_id
#         ordinal_id += 1
#     return actual_to_ordinal_mapping
#
# def update_distance_matrix(distance_matrix, actual_to_ordinal_mapping):
#     updated_distance_matrix = {}
#     for actual_id, distances in distance_matrix.items():
#         ordinal_id = actual_to_ordinal_mapping[actual_id]
#         updated_distance_matrix[ordinal_id] = distances
#     return updated_distance_matrix
#
# def find_closest_neighbors(distance_matrix, threshold=0.3):
#     closest_neighbors = {}
#     for bs_id, distances in distance_matrix.items():
#         closest_neighbors[bs_id] = [i+1 for i, dist in enumerate(distances) if dist < threshold]
#     return closest_neighbors
#
# # Read distance matrix from CSV
# csv_file = 'D:/PycharmProjects/jaffarDRL/distance_matrix.csv'  # Replace with your CSV file path
# distance_matrix = read_distance_matrix_from_csv(csv_file)
#
# # Map actual IDs to ordinal IDs
# actual_to_ordinal_mapping = map_actual_to_ordinal_ids(distance_matrix)
#
# # Update distance matrix with ordinal IDs
# distance_matrix = update_distance_matrix(distance_matrix, actual_to_ordinal_mapping)
#
# # Find closest neighbors
# closest_neighbors = find_closest_neighbors(distance_matrix)
#
# # Print closest neighbors
# print("Closest Neighbors:")
# total = 0
# for bs_id, neighbors in closest_neighbors.items():
#     total += len(neighbors)
#     print(f"Base Station {bs_id}: {(neighbors)} : {len(neighbors)} : {math.pow(132, (132-len(neighbors)))}")
#
# print("math.pow(132, 132) - total", (math.pow(132, 132) - total))
# print("math.pow(132, 132) - total", total)
# print(math.pow(2, 132))
from itertools import product


def action_space(num_base_stations):
    base_station_ids = [i for i in range(1, 3)]

    result = list(product(base_station_ids, repeat=132))

    print(len(result))
    print(result)

action_space(132)