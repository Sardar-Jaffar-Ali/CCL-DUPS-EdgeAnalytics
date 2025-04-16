import numpy as np
import pandas as pd
import torch

from base_station import BaseStation


class Environment:
    def __init__(self, num_bs=132, max_du_capacity=100, distance_matrix=None, initial_traffic=None):
        self.num_bs = num_bs
        self.all_traffic_df = None
        self.eNodeB_IDs = None
        self.max_du_capacity = max_du_capacity
        self.distance_matrix = distance_matrix
        self.initial_traffic = initial_traffic
        self.date = 0
        self.time = 0
        self.actions = None
        self.bs_vector = {}
        self.load_distance_matric()
        self.predicted_traffic()
        self.load_traffic_data()
        self.create_bs_vector()
        self.penalty = 0.0

    def load_distance_matric(self):
        distance_matrix_df = pd.read_csv('distance_matrix.csv', index_col=0)
        col_names = distance_matrix_df.columns.tolist()
        distance_matrix_df.columns = col_names
        distance_matrix_df.index = col_names
        self.distance_matrix = distance_matrix_df  # = distance_matrix

    def get_distance_vector(self, bs_id):
        # print(self.distance_matrix)
        # print(f"My bs id is: {bs_id} -- type is: {type(bs_id)}")
        # print(self.distance_matrix.columns)
        # print(self.distance_matrix.index)
        return self.distance_matrix.loc[bs_id]

    def predicted_traffic(self, date_x=0, time_y=0):
        df = pd.read_csv('DRL.csv')  # Update 'your_file.csv' with the actual file path
        # Choose Date (X) and Time (Y)
        date_x = date_x
        time_y = time_y  # For example, using time index 5, update as needed

        # Filter dataframe for the specified Date and Time
        filtered_df = df[(df['Date'] == date_x) & (df['Time'] == time_y)]

        # Get unique eNodeBs
        unique_enodebs = filtered_df['eNodeB'].unique()

        # Create arrays for Total Traffic and Predicted Traffic
        initial_traffic = np.zeros(len(unique_enodebs))
        next_traffic = np.zeros(len(unique_enodebs))

        # Populate arrays based on unique eNodeBs
        for i, enodeb in enumerate(unique_enodebs):
            initial_traffic[i] = filtered_df[filtered_df['eNodeB'] == enodeb]['Total Traffic'].values[0]
            next_traffic[i] = filtered_df[filtered_df['eNodeB'] == enodeb]['Predicted Traffic'].values[0]
        # Now, total_traffic_array and predicted_traffic_array contain the values for Total Traffic and Predicted Traffic for the specified Date and Time

        # print("Date value is: ", date_x)
        # print(f"Data for day: {date_x} --- time:{time_y} --- is: {initial_traffic}")
        return initial_traffic, next_traffic

    def load_traffic_data(self):
        df = pd.read_csv('DRL.csv')  # Update 'your_file.csv' with the actual file path
        self.all_traffic_df = df
        self.eNodeB_IDs = df['eNodeB'].unique().tolist()

    def set_date_and_time(self, date, time):
        if self.date >= 6:
            self.date = 0
        else:
            self.date = date

        self.time = time

    def load_eNodeB_data(self, base_station):
        filtered_df = self.all_traffic_df[(self.all_traffic_df['eNodeB'] == base_station)]
        grouped_df = filtered_df.groupby('Date')
        date_idx_dict = {}
        bs_data_dict = {}
        for day in list(grouped_df.groups.keys()):
            date_idx_dict[day] = filtered_df.loc[grouped_df.groups[day]]
            time_indices = date_idx_dict[day].groupby('Time')
            time_indexed_data = {}
            for time_ in list(time_indices.groups.keys()):
                initial_traffic = \
                    date_idx_dict[day][date_idx_dict[day]['Time'] == time_]['Total Traffic'].values[0]
                next_traffic = \
                    date_idx_dict[day][date_idx_dict[day]['Time'] == time_]['Predicted Traffic'].values[0]
                time_indexed_data[time_] = [initial_traffic, next_traffic]

            bs_data_dict[day] = time_indexed_data

        return bs_data_dict

    def create_bs_vector(self):
        # initial_traffic, next_traffic = self.predicted_traffic()
        for i in self.eNodeB_IDs:
            traffic = self.load_eNodeB_data(base_station=i)
            bs = BaseStation(bs_id=str(i), env=self, bs_traffic=traffic)
            # bs.set_traffic(initial_traffic[i], next_traffic[i])
            self.bs_vector[i] = bs

        # print(list(self.bs_vector.keys()))

    # def step(self, time_y, actions):
    #     self.actions = [action.item() for action in actions]
    #     # print([action.item() for action in actions])
    #     done = [False] * self.num_bs
    #     if time_y == 23:
    #         if self.date == 7:
    #             self.date = 0
    #         else:
    #             self.date += 1
    #
    #         done = [True] * self.num_bs
    #
    #     initial_traffic, next_traffic = self.predicted_traffic(date_x=self.date, time_y=time_y)
    #     # print("Date is: ", self.date, " time is: ", time_y)
    #     for i in range(len(next_traffic)):
    #         base_station = self.bs_vector[i]
    #         # print("Traffic is: ", next_traffic)
    #         # print("Index is: ", i, " Length is: ", len(next_traffic))
    #         # base_station.predicted_traffic = next_traffic[i]
    #         # for x in range(len(next_traffic)):
    #         #     print("index: {} -- Value: {}".format(x, next_traffic[x]))
    #         # print("Newly loaded traffic: ", initial_traffic)
    #         base_station.switch_du(status=actions[i].item(), new_traffic=initial_traffic, day=self.date)
    #
    #     rewards = []
    #     for i in self.eNodeB_IDs:
    #         energy_for_du, total_merged = self.get_consumed_energy(bs_id=self.bs_vector[i].bs_id)
    #         latency_violation = self.check_latency_violation(self.get_latency(bs_id=self.bs_vector[i].bs_id))
    #         reward = self.calculate_reward(energy_consumption=energy_for_du, latency=latency_violation)
    #         # if total_merged > 100:
    #         #     reward += 5
    #
    #         rewards.append(reward)
    #
    #     return self.get_state(), rewards, done
    def counts(self, actions):
        if isinstance(actions, int):
            # If actions is an integer, return 0 for both turned off and turned on
            return 0, 0

        turn_off_ctr = sum(1 for a in actions if a == 0)
        turn_on_ctr = sum(1 for a in actions if a == 1)

        return turn_off_ctr, turn_on_ctr
        # return turn_on_ctr

    def step(self, time_y, actions):
        # print("Length of actions:", len(actions))
        # print("AAA", actions)
        # print("Number of base stations:", len(self.bs_vector))

        # if isinstance(actions, int):
        #     print("Actions is an integer:", actions)
        #     return None, None, None

        # Assuming actions is a list of length 132
        self.actions = [torch.tensor(action).item() for action in actions]
        done = [False] * len(self.eNodeB_IDs)
        total_traffic_before_action = sum(self.additional_metrics())
        for i in range(len(self.eNodeB_IDs)):
            base_station = self.bs_vector[self.eNodeB_IDs[i]]
            # print("Current action index:", i)
            # Access the action for the current base station
            action_for_bs = actions[i]
            base_station.switch_du(status=int(action_for_bs), day=self.date, time=self.time)

        rewards = []
        # print("actionaa",actions)
        total_turned_off, total_turned_on = self.counts(actions)
        print("Total turned off:", total_turned_off)
        print("Total turned on:", total_turned_on)

        # total_turned_on = self.counts(actions) * 100
        difference = (total_turned_on*100) - total_traffic_before_action
        # print(f"____{total_turned_on} ____{total_traffic_before_action}")
        penalty = self.deficit_penalty(difference, total_traffic_before_action)
        self.penalty = penalty

        for i in self.eNodeB_IDs:
            energy_for_du, total_merged = self.get_consumed_energy(bs_id=self.bs_vector[i].bs_id)
            latency = self.latency_normalization(self.get_latency(bs_id=self.bs_vector[i].bs_id))
            reward = self.calculate_reward(energy_consumption=energy_for_du, latency=latency)
            if penalty > 0.2:
                reward -= (0.2 * penalty)
            elif penalty <= 0:
                reward += (0.2 * penalty)
            elif 0 < penalty < 0.2:
                reward += (0.2 * penalty)
            rewards.append(reward)

        print("time is: ", self.time)
        if self.time == 23:

            done = [True] * len(self.eNodeB_IDs)

        return self.get_state(), rewards, done

    # def step(self, time_y, actions):
    #     self.actions = [action.item() for action in actions]
    #     done = [False] * len(self.eNodeB_IDs)
    #     initial_traffic, next_traffic = self.predicted_traffic(date_x=self.date, time_y=time_y)
    #     total_traffic_before_action = sum(self.additional_metrics())
    #
    #     previous_rewards = []  # Store previous rewards
    #     for i in self.eNodeB_IDs:
    #         energy_for_du, total_merged = self.get_consumed_energy(bs_id=self.bs_vector[i].bs_id)
    #         latency = self.latency_normalization(self.get_latency(bs_id=self.bs_vector[i].bs_id))
    #         reward = self.calculate_reward(energy_consumption=energy_for_du, latency=latency)
    #
    #         previous_rewards.append(reward)  # Store previous rewards
    #
    #     for i in range(len(self.eNodeB_IDs)):
    #         base_station = self.bs_vector[self.eNodeB_IDs[i]]
    #         base_station.switch_du(status=actions[i].item(), day=self.date, time=self.time)
    #
    #     if self.time == 23 and self.date == 6:
    #         print("Resetting date.........................")
    #         self.date = 0
    #
    #     rewards = []
    #     total_turned_on = self.counts(actions) * 100
    #     difference = total_turned_on - total_traffic_before_action
    #     penalty = self.deficit_penalty(difference, total_traffic_before_action)
    #
    #     if 0 <= penalty <= 2.2:
    #         return self.get_state(), previous_rewards, done
    #
    #     for i in self.eNodeB_IDs:
    #         energy_for_du, total_merged = self.get_consumed_energy(bs_id=self.bs_vector[i].bs_id)
    #         latency = self.latency_normalization(self.get_latency(bs_id=self.bs_vector[i].bs_id))
    #         reward = self.calculate_reward(energy_consumption=energy_for_du, latency=latency)
    #         previous_rewards.append(reward)  # Store previous rewards
    #
    #         if penalty > 0:
    #             reward = -(0.5 * penalty)
    #         else:
    #             reward += (0.5 * penalty)
    #
    #         rewards.append(reward)
    #
    #     if self.time >= 23:
    #         done = [True] * len(self.eNodeB_IDs)
    #
    #     return self.get_state(), rewards, done
    def get_state(self):
        observations = []
        for i in self.eNodeB_IDs:
            # self.date = self.date % 7
            # temp_vector = [self.time, self.bs_vector[i].du_status,
            #                self.bs_vector[i].existing_traffic, self.bs_vector[i].predicted_traffic]
            # self.bs_vector[i].traffic[int(self.date)][int(self.time)][1]]
            temp_vector = [self.time, self.bs_vector[i].du_status,
                           self.bs_vector[i].existing_traffic,
                           self.bs_vector[i].predicted_traffic]

            # print(temp_vector)
            observations.append(temp_vector)

        observations = torch.tensor(observations)
        # print("observation is: ", observations)
        return observations

    def get_consumed_energy(self, bs_id):
        du_energy_consumption = 5  # Joules..
        bs = self.bs_vector[int(bs_id)]
        idle_energy = 50
        total_merged = 0.0
        if bs.du_status == 1:
            total_merged = bs.existing_traffic
            total_energy_consumed = (du_energy_consumption * total_merged) + idle_energy
            normalized = self.value_scaling(value=total_energy_consumed, max_x=600, min_x=0)

            return normalized, total_merged
        else:
            return self.value_scaling(value=idle_energy, max_x=600, min_x=0), 0

    # def get_consumed_energy(self):
    #     total_energy = 0.0
    #     energy_per_du = 10  # Joules..
    #     turned_on_count = 0
    #     du_pool_energy_consumption = []
    #     for key, bs in self.bs_vector.items():
    #         if bs.du_status == 1:
    #             turned_on_count += 1
    #             total_energy += energy_per_du
    #             du_pool_energy_consumption.append(energy_per_du)
    #         else:
    #             du_pool_energy_consumption.append(0)
    #
    #     print(f"Total number of active DUs: {turned_on_count} --- Total energy consumed: {du_pool_energy_consumption}")
    #     return du_pool_energy_consumption

    # def get_latency(self, bs_id):
    #     latency_per_gb = 1  # ms
    #     latencies = []
    #     bs_keys = list(self.bs_vector.keys())
    #     for i in range(len(bs_keys)):
    #         latency = latency_per_gb * self.bs_vector[bs_keys[i]].initial_traffic
    #         if latency > 0.0:
    #             latencies.append(latency)
    #         else:
    #             latencies.append(0.0)
    #     return latencies

    def get_latency(self, bs_id):
        return self.bs_vector[int(bs_id)].latency
        # latency = latency_per_gb_coefficient * target_bs.initial_traffic

        # print(f"Distance from BS: {target_bs.bs_id} to other BSs: {self.get_distance_vector(target_bs.bs_id)}")
        #
        # if latency > 0.0:
        #     return latency
        # else:
        #     return 0

        # return

    def value_scaling(self, value, max_x, min_x):
        max_range = 1
        min_range = 0
        scaled = round((max_range - min_range) * ((value - min_x) / (max_x - min_x)) + min_range, 2)
        return scaled

    def compute_advantage(self):
        num_off_du = 0
        for key, bs in self.bs_vector.items():
            if bs.du_status == 0:
                num_off_du += 1

        max_x = 131
        min_x = 1
        advantage = self.value_scaling(max_x=max_x, min_x=min_x, value=num_off_du)

        return advantage

    #
    def compute_fronthaul_latency(self, distance_to_merging_pi):
        # if by minimizing 1s results in beyond distance , 10 km or 75 us (5us/km means 50us for propagation delay + 25us of misc. delay)
        coefficient = 5  # us/km
        fronthaul_delay = coefficient * distance_to_merging_pi
        return fronthaul_delay

    def compute_processing_delay(self, available_traffic):
        coefficient = 0.02  # microseconds per GB
        return coefficient * available_traffic

    def additional_metrics(self):
        total_current_traffic = []
        for key, bs in self.bs_vector.items():
            total_current_traffic.append(bs.specific_time_traffic(self.date, self.time))

        return total_current_traffic

    def latency_normalization(self, latency):
        max_x = 75
        min_x = 0

        min_capacity_threshold = 0
        violation_threshold = 10
        if latency <= min_capacity_threshold:
            # print(f"Traffic: {latency} --- Penalty: {0}")
            return 0
        elif latency > violation_threshold:
            # print(f"Traffic: {latency} --- Penalty: {10}")
            return 10
        else:
            penalty = self.value_scaling(max_x=max_x, min_x=min_x, value=latency)
            # print(f"Traffic: {latency} --- Penalty: {penalty}")
            return penalty

    def check_latency_violation(self, latency):
        max_x = 100
        min_x = 0

        min_capacity_threshold = 0
        violation_threshold = 10
        if latency <= min_capacity_threshold:
            # print(f"Traffic: {latency} --- Penalty: {0}")
            return 0
        elif latency > violation_threshold:
            # print(f"Traffic: {latency} --- Penalty: {10}")
            return 10
        else:
            penalty = self.value_scaling(max_x=max_x, min_x=min_x, value=latency)
            # print(f"Traffic: {latency} --- Penalty: {penalty}")
            return penalty

    def check_capacity_violation_penalty(self):
        total_used_capacity = 0.0
        for key, bs in self.bs_vector.items():
            if bs.du_status == 1:
                if bs.existing_traffic > 100:
                    total_used_capacity += bs.existing_traffic

        max_x = 100
        min_x = 0

        min_capacity_threshold = 90
        violation_threshold = 100
        if total_used_capacity <= min_capacity_threshold:
            # print(f"Traffic: {total_used_capacity} --- Penalty: {0}")
            return 0
        elif total_used_capacity > violation_threshold:
            # print(f"Traffic: {total_used_capacity} --- Penalty: {10}")
            return 10
        else:
            penalty = self.value_scaling(max_x=max_x, min_x=min_x, value=total_used_capacity)
            # print(f"Traffic: {total_used_capacity} --- Penalty: {penalty}")
            return penalty

    def scaling_variant(self,  value, max_x, min_x):
        max_range = 10
        min_range = -10
        scaled = round((max_range - min_range) * ((value - min_x) / (max_x - min_x)) + min_range, 2)
        return scaled

    def deficit_penalty(self, difference, total_traffic_before_action):
        max_deficit = -13200
        max_excess = 13200
        penalty = self.scaling_variant(max_x=max_excess, min_x=max_deficit, value=difference)
        print(f"Total traffic: {total_traffic_before_action} --- Traffic difference: {difference} --- Penalty: {penalty}")
        return penalty

    def calculate_reward(self, energy_consumption, latency):
        # if the number of turn-off DUs are more (Minimizing 1s in the action), it gets higher reward
        # if by minimizing 1s results in to beyond capacity, it gets penalty
        # if by minimizing 1s results in beyond distance , 10 km or 75 us (5us/km means 50us for propagation delay + 25us of misc. delay)
        # alpha = 0.2
        # reward = alpha * energy_consumption + (1 - alpha) * latency

        reward = 0
        # if latency > 0:
        # reward = (- energy_consumption - latency) - self.check_capacity_violation_penalty()
        reward = +(0.3 * self.compute_advantage()) - (0.3 * latency) - (0.2 * self.check_capacity_violation_penalty())
        # reward += self.compute_advantage()
        #     return reward
        # else:
        #     reward = -(0.1 * energy_consumption) - (0.2 * latency) - (0.2 * self.check_capacity_violation_penalty())
        #     reward -= self.check_capacity_violation_penalty()
        # reward += self.compute_advantage()
        return reward
