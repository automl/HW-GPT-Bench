import pickle
import numpy as np
i = 0
import os
increment = 2500
devices = ["a6000","a100","v100", "helix_cpu"] #["P100","a6000", "rtx2080", "rtx3080", "v100", "v100", "a40", "h100", "cpu_mlgpu", "cpu_alldlc", "cpu_p100", "cpu_p100", "cpu_a6000", "cpu_meta"]
suffix = ["", "_m", "_l"]
restart = []
for device in devices:
 for s in suffix:
  cat_list = []
  for i in range(0, 10000, increment):
   path = "latency_" + device + s + "/efficiency_energy_observations_tracker_" + str(i) + "_" + str(i+increment) + ".pkl"
   if not os.path.exists(path):
     if "cpu" not in device and device+s not in restart:
      restart.append(device + s)
     continue
   with open(path,"rb") as f:
      a = pickle.load(f)
      cat_list.extend(a)
  a = cat_list
  unique = []
  for i in a:
    if i not in unique:
        unique.append(i)
  print("For device: ", device + s)
  print(len(unique))
  # to restart
  if len(unique) < 10000 and "cpu" not in device and device+s not in restart:
      restart.append(device + s)
print("Restart: ", restart)
devices = ["a100","v100","helix_cpu"]  #["P100","a6000", "rtx2080", "rtx3080", "v100", "v100", "a40", "h100", "cpu_mlgpu", "cpu_alldlc", "cpu_p100", "cpu_p100", "cpu_a6000", "cpu_meta"]
suffix = ["", "_m", "_l"]
restart = []
for device in devices:
 for s in suffix:
  cat_list = []
  for i in range(0, 10000, increment):
   path = "latency_" + device + s + "/efficiency_observations_" + str(i) + "_" + str(i+increment) + ".pkl"
   if not os.path.exists(path):
      restart.append(device + s)
      continue
   with open(path,"rb") as f:
      a = pickle.load(f)
      cat_list.extend(a)
  a = cat_list
  unique = []
  for i in a:
    if i not in unique:
        unique.append(i)
  print("For device: ", device + s)
  print(len(unique))
  # to restart
  if len(unique) < 10000:
      restart.append(device + s)
print("Restart: ", restart)
