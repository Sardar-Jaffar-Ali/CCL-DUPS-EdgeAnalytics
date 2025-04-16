# import numpy as np
# import pandas as pd
#
#
# # def predicted_traffic(date_x=0, time_y=0):
# #     df = pd.read_csv('DRL.csv')  # Update 'your_file.csv' with the actual file path
# #     # Choose Date (X) and Time (Y)
# #     date_x = date_x
# #     time_y = time_y  # For example, using time index 5, update as needed
# #
# #     # Filter dataframe for the specified Date and Time
# #     filtered_df = df[(df['Date'] == date_x) & (df['Time'] == time_y)]
# #
# #     # Get unique eNodeBs
# #     unique_enodebs = filtered_df['eNodeB'].unique()
# #
# #     # Create arrays for Total Traffic and Predicted Traffic
# #     initial_traffic = np.zeros(len(unique_enodebs))
# #     next_traffic = np.zeros(len(unique_enodebs))
# #
# #     # Populate arrays based on unique eNodeBs
# #     for i, enodeb in enumerate(unique_enodebs):
# #         initial_traffic[i] = filtered_df[filtered_df['eNodeB'] == enodeb]['Total Traffic'].values[0]
# #         next_traffic[i] = filtered_df[filtered_df['eNodeB'] == enodeb]['Predicted Traffic'].values[0]
# #     # Now, total_traffic_array and predicted_traffic_array contain the values for Total Traffic and Predicted Traffic for the specified Date and Time
# #     return initial_traffic, next_traffic, unique_enodebs
#
# # days = 5
# # max_time = 24
# # valid_length = 132
# # missing_days = {}
# #
# # valid_enodebs = [1, 2, 6, 8, 12, 13, 16, 18, 21, 28, 29, 44, 46, 48, 58, 59, 63, 67, 69, 107, 118, 122, 133, 140, 157, 162, 163, 186, 201, 209, 210, 215, 218, 221, 243, 255, 278, 316, 320, 326, 334, 335, 336, 341, 346, 347, 351, 352, 353, 356, 359, 362, 375, 402, 413, 421, 422, 423, 425, 429, 436, 441, 473, 481, 488, 506, 510, 524, 536, 540, 552, 554, 564, 565, 570, 610, 615, 631, 632, 655, 661, 663, 673, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 716, 727, 728, 742, 765, 766, 772, 773, 775, 776, 785, 820, 826, 832, 847, 883, 902, 906, 907, 923, 970, 1017, 1018, 1024, 1025, 1028, 1031, 1035, 1040, 1045, 1046, 1048, 1050, 1053, 1061, 1062]
# #
# # for day in range(days):
# #     missing = []
# #     for time in range(max_time):
# #         print(f"====== Day {day} ---- Time: {time} -----")
# #         init_traffic, x, enode_b = predicted_traffic(date_x=day, time_y=time)
# #         for enb in valid_enodebs:
# #             if enb not in enode_b:
# #                 missing.append(enb)
# #
# #         missing_days[time] = missing
# #
# #         # if len(init_traffic) < valid_length:
# #         #     missing_days[day] :
# #
# # print(missing_days)
#
# def predicted_traffic(base_station=1):
#     df = pd.read_csv('DRL.csv')  # Update 'your_file.csv' with the actual file path
#     # Choose Date (X) and Time (Y)
#     # Filter dataframe for the specified Date and Time
#     filtered_df = df[(df['eNodeB'] == base_station)]
#
#     # print(filtered_df.to_string())
#     # Get unique eNodeBs
#     unique_enodebs = filtered_df['eNodeB'].unique()
#     print("Base station IDs: ", df['eNodeB'].unique())
#     return filtered_df
#
#
# def construct_dict(dataframe):
#     data_dict = {}
#     date_indexed_dict = {}
#
#     grouped_df = dataframe.groupby('Date')
#     # grouped_by_time = dataframe.loc[grouped_df.groups[0]].groupby('Time')
#     # print(grouped_by_time.groups.keys())
#     # for time in list(grouped_by_time.groups.keys()):
#     #     time_indexed_dict[time] = dataframe.loc[grouped_by_time.groups[time]]
#     #
#     # for key, value in time_indexed_dict.items():
#     #     print(f"Time: {key} \n {'--' * 5} \n {value}")
#     #
#     test_dict = {}
#     for day in list(grouped_df.groups.keys()):
#         date_indexed_dict[day] = dataframe.loc[grouped_df.groups[day]]
#         time_indices = date_indexed_dict[day].groupby('Time')
#         time_indexed_data = {}
#         for time_ in list(time_indices.groups.keys()):
#             initial_traffic = date_indexed_dict[day][date_indexed_dict[day]['Time'] == time_]['Total Traffic'].values[0]
#             next_traffic = date_indexed_dict[day][date_indexed_dict[day]['Time'] == time_]['Predicted Traffic'].values[0]
#             # print(f"Day: {day}, time: {time_}, total: {initial_traffic}, pred: {next_traffic}")
#             # next_traffic = date_indexed_dict[day][date_indexed_dict[day]['Time'] == time_]['Predicted Traffic']
#             time_indexed_data[time_] = [initial_traffic, next_traffic]
#
#             # print(date_indexed_dict[day][date_indexed_dict[day]['Time'] == time_].to_string())
#         test_dict[day] = time_indexed_data
#
#     # print(test_dict[0])
#     for key, value in test_dict.items():
#         print(f"Day: {key} \n {'--' * 5} \n {value}")
#         # time_indices = date_indexed_dict[day].groupby('Time')
#         # for time_ in list(time_indices.groups.keys()):
#         #     print(dataframe.loc[time_indices.groups[day][time_]])
#         # temp_dict = {}
#         # for t_ in list(time_indices.groups.keys()):
#         #    time_indices.groups.values()
#         # test_dict[day] = {time_indices.groups.keys() : }
#
#     # for key, value in test_dict.items():
#     #     print(f"Day: {key} \n {'--' * 5} \n {value}")
#
#
# filtered_df = predicted_traffic(base_station=1)
#
# construct_dict(filtered_df)

# value = [i for i in range(23)]
# print(value)