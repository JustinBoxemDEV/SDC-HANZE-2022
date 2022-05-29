# scuffed script to sort csv by image name

import pandas as pd
import csv

csv_file_path = "D:/SDC/sdc_data/justin_data/original/100_new_data images 30-03-2022 15-17-40.csv"

actions_frames = pd.read_csv(csv_file_path)
csv_array = []
final_array = []

# count the amount of lines in the csv
with open(csv_file_path) as f:
    row_count = sum(1 for line in f)

for i in range(row_count-1):
    # print(i)
    csv_array.append([actions_frames.iloc[i, 0], actions_frames.iloc[i, 1], actions_frames.iloc[i, 2], actions_frames.iloc[i, 3]])

csv_array.sort(key=lambda x: x[-1])

out_path = "D:/SDC/sdc_data/justin_data/original/sorted_100_new_data images 30-03-2022 15-17-40.csv"

with open(out_path, 'w', newline='') as out_file:
    out_file_writer = csv.writer(out_file)

    for value in csv_array:
        out_file_writer.writerow(value)
