import pickle
i = 0
increment = 2500
cat_list = []
for i in range(0, 10000, increment):
 path = "latency_a100/efficiency_observations_" + str(i) + "_" + str(i+increment) + ".pkl"
 with open(path,"rb") as f:
    a = pickle.load(f)
    cat_list.extend(a)
a = cat_list