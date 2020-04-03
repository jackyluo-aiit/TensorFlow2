import csv
import os


def write_history(file_path, trial, acc, reward, state):
    with open(file_path, mode='a+') as f:
        data = [trial, acc, reward]
        data.extend(state)
        writer = csv.writer(f)
        writer.writerow(data)


def remove_history(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)
