import heapq
import itertools

import numpy as np
import pandas as pd


class BaseStation:
    def __init__(self, bs_id, env, bs_traffic, max_du_capacity=100):
        self.bs_id = bs_id
        self.du_capacity = max_du_capacity
        self.du_status = 1
        self.traffic = bs_traffic
        self.existing_traffic = self.traffic[0][0][0]
        self.environment = env
        self.predicted_traffic = self.traffic[0][0][1]
        self.energy_per_gigabit = 10  # 10 joules/unit
        self.latency = 0.0
        self.metrics_dict = []

    def __repr__(self):
        return "BS ID: {} -- DU Status: {}".format(self.bs_id, self.du_status)

    # def compute_energy_consumption(self):
    #     return self.initial_traffic * self.energy_per_gigabit

    def set_traffic(self, initial, predicted):
        self.existing_traffic = initial
        self.predicted_traffic = predicted

        # update indices for tracking date and time in the traffic file (loading)

    def update_bs_traffic(self, new_traffic):
        self.existing_traffic += new_traffic

    def get_nearest_bs(self):
        # this comes from the distance matrix vector
        # print(f"My bs id is: {self.bs_id} -- type is: {type(self.bs_id)}")
        distance_matrix = self.environment.get_distance_vector(str(self.bs_id))
        # print("---------------", distance_matrix.shape)
        selected_indices = [index for index, value in enumerate(distance_matrix) if float(value) < 0.5]
        # print(selected_indices)
        # print(f"Selected list for base station ID {self.bs_id} : {len(selected)}")

        # second_min_value = heapq.nsmallest(2, distance_matrix)[-1]
        #
        # nearest_bs = np.where(distance_matrix == second_min_value)[0][0]
        # return nearest_bs
        return selected_indices

    def check_bs_capacity(self):
        return self.du_capacity - self.predicted_traffic

    def is_on(self):
        return True if self.du_status == 1 else False

    def check_available_capacity(self, bs_id):
        """
        Check total available capacity in %ntage.
        :param bs_id:
        :return: percentage of utilized capacity
        """
        maximum_capacity = 100  # Gigabytes
        occupied_capacity = self.environment.bs_vector[bs_id].existing_traffic
        utilized_percentage = (occupied_capacity / 100.0) * maximum_capacity
        return utilized_percentage

    # def specific_time_traffic(self, actual_day, time):
    #     current_traffic = self.traffic[actual_day][time][0]
    #     return current_traffic
    def specific_time_traffic(self, actual_day, time):
        # Get traffic data for a specific time on a given day with cyclic behavior for days beyond day 6

        actual_day = actual_day % 7  # Calculate the actual day within the range of 0 to 6

        if actual_day not in self.traffic:
            # Handle cases where the key 'actual_day' is not present in self.traffic
            # You can choose how to handle this scenario based on your requirements
            return None  # Or any other appropriate action

        if time not in self.traffic[actual_day]:
            # Handle cases where the key 'time' is not present in self.traffic[actual_day]
            # You can choose how to handle this scenario based on your requirements
            return None  # Or any other appropriate action

        current_traffic = self.traffic[actual_day][time][0]
        return current_traffic

    def move_traffic_to_nearest_bs(self, nearest_bs, new_traffic, day):
        bs_dist_status_dict = {}
        for idx in nearest_bs:
            actual_id = self.environment.eNodeB_IDs[idx]
            if self.environment.actions[idx] == 1:
                bs_dist_status_dict[idx] = [
                    self.environment.get_distance_vector(self.bs_id)[str(actual_id)],
                    self.environment.bs_vector[actual_id].existing_traffic,
                    self.environment.actions[idx]
                ]
            else:
                pass

        target_bs = dict(sorted(bs_dist_status_dict.items(), key=lambda item: item[1]))
        for key in list(target_bs.keys()):
            max_allowed_capacity = 90  # percent
            if self.check_available_capacity(self.environment.eNodeB_IDs[key]) > max_allowed_capacity:
                del target_bs[key]

        if len(target_bs) > 0:
            actual_bs_id = self.environment.eNodeB_IDs[list(target_bs.keys())[0]]
            dest_bs_object = self.environment.bs_vector[actual_bs_id]

            distance = target_bs[list(target_bs.keys())[0]][0]
            traffic = target_bs[list(target_bs.keys())[0]][1]

            fronthaul_latency = self.environment.compute_fronthaul_latency(distance)
            processing_latency = self.environment.compute_processing_delay(traffic)
            self.existing_traffic = -0.0
            my_traffic = new_traffic[0]
            total_new_traffic = my_traffic
            dest_bs_object.update_bs_traffic(total_new_traffic)
            self.predicted_traffic = new_traffic[1]

            self.latency = fronthaul_latency + processing_latency
        else:
            self.existing_traffic = new_traffic[0]
            self.predicted_traffic = new_traffic[1]

            processing_latency = self.environment.compute_processing_delay(self.existing_traffic)
            self.latency = processing_latency

            # pass
            # print("++++++++++++++++++++++++ Traffic for BS: {} could not be moved to BS: {}".format(self.bs_id, dest_bs_object.bs_id))
    #
    # def switch_du(self, status, day, time):
    #     # Implement decision, changing DU status to either on/off
    #     self.du_status = status
    #     if status == 0:
    #         nearest_bs = self.get_nearest_bs()
    #         self.move_traffic_to_nearest_bs(nearest_bs, new_traffic, day)
    #     else:
    #         pass
    #     # 1. If action is switch off (0), then move its traffic to nearest BS ( i.e., dist <= 0.5)
    #     # 2.

    # def switch_du(self, status, day, time):
    #     # Implement decision, changing DU status to either on/off
    #     # load traffic for day=day and time=time
    #     new_traffic = self.traffic[day][time][0]
    #
    #     # print(f"Day: {day} --- Time: {time} --- {new_traffic} --- BS ID: {self.bs_id}")
    #     self.du_status = status
    #     if status == 0:
    #         nearest_bs = self.get_nearest_bs()
    #         self.move_traffic_to_nearest_bs(nearest_bs, new_traffic, day)
    #     else:
    #         pass
        # 1. If action is switch off (0), then move its traffic to nearest BS ( i.e., dist <= 0.5)
        # 2.

    def switch_du(self, status, day, time):
        # Implement decision, changing DU status to either on/off
        # Load traffic for day=day and time=time. For days beyond day 6, cycle back to day 0 data.

        actual_day = day % 7  # Calculate the actual day within the range of 0 to 6

        if actual_day not in self.traffic:
            # Handle cases where the key 'actual_day' is not present in self.traffic
            # You can choose how to handle this scenario based on your requirements
            return None  # Or any other appropriate action

        if time not in self.traffic[actual_day]:
            # Handle cases where the key 'time' is not present in self.traffic[actual_day]
            # You can choose how to handle this scenario based on your requirements
            return None  # Or any other appropriate action

        new_traffic = [self.traffic[actual_day][time][0], self.traffic[actual_day][time][1]]


        self.du_status = status
        if status == 0:
            nearest_bs = self.get_nearest_bs()
            self.move_traffic_to_nearest_bs(nearest_bs, new_traffic, actual_day)
        else:
            self.existing_traffic = self.traffic[actual_day][time][0]
            self.predicted_traffic = self.traffic[actual_day][time][1]