import os
import csv


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


def load_data2(path):
    csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    input_file = os.path.join(path)
    en = []
    fr = []
    count = 0
    with open(input_file, "r") as f:
        data = csv.reader(f, 'mydialect')
        for line in data:
            try:
                if line[0] != '' and line[1] != '':
                    en.append(line[0])
                    fr.append(line[1])
                    count += 1
            except:
                continue
        print(count)
    csv.unregister_dialect('mydialect')
    return en, fr
