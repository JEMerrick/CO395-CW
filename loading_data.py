import numpy as np

#how to split this into random 80/10/10 ratio?
#can't select last 10% as they will all be in room 4, not enough variation
all = np.loadtxt("WIFI.db/clean_dataset.txt")
print(all)

source_1 = []
source_2 = []
source_3 = []
source_4 = []
source_5 = []
source_6 = []
source_7 = []
room_no = []

for datapoint in all:
    source_1.append(datapoint[0]) #all[datapoint][0]
    source_2.append(datapoint[1])
    source_3.append(datapoint[2])
    source_4.append(datapoint[3])
    source_5.append(datapoint[4])
    source_6.append(datapoint[5])
    source_7.append(datapoint[6])
    room_no.append(datapoint[7])

for signal in source_1:
    print(signal)

for signal in room_no:
    print(signal)
