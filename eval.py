# ============================================================== #
# imports {{{
# ============================================================== #
import gym
import sys
from gym import spaces
import numpy as np
import scipy.io as sio
import json
import os
import re
import ray
import ray.rllib.algorithms.apex_dqn.apex_dqn as dqn
import pickle
from CodeEnv import *
import ctypes
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ctypes.windll.shell32.IsUserAnAdmin()

# }}}
# ============================================================== #
# parameters {{{
# ============================================================== #
tmpdir = r"\MYP\fc94"

dB_range = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
minCwErr = 200
maxcw = 1000000
NUM = 1

path = r"\MYP\fc94\RL_SOURCE\ray_results\CodeEnv\APEX_CodeEnv_5c856_00000_0_2023-02-01_12-26-48"
SAVE = True
path1 = path + r"\params.json"

if SAVE:
    save_path = path + "/res_{}.mat".format(NUM)
    save_path_txt = path + "/res_{}.txt".format(NUM)

# }}}
# ============================================================== #

with open(path + r"\params.pkl", mode='rb') as h:
    config = pickle.load(h)
"""""with open(path1) as h:    config = json.loads(h.read())"""""
print(config)
env_config = config["env_config"]

# find all checkpoint and load the latest
filenames = os.listdir(path)
checkpoint_numbers = []
for fn in filenames:
    m = re.findall('checkpoint_(\d+)', fn)
    if not m: continue
    print(m[0])
    checkpoint_numbers.append(m[0])

mc = max(checkpoint_numbers)
checkpoint_path = path + "/" + "checkpoint_{}".format(mc, mc)
print("found {} checkpoints".format(len(checkpoint_numbers)))
print("restoring " + checkpoint_path)

# ============================================================== #
# evaluation {{{
# ============================================================== #
# ray.init()
ray.init()

trainer = dqn.ApexDQN(config=config, env=CodeEnv)

trainer.restore(checkpoint_path)
env = CodeEnv(env_config)
n = env.n

dB_len = len(dB_range)
BitErr = np.zeros([dB_len], dtype=int)
CwErr = np.zeros([dB_len], dtype=int)
totCw = np.zeros([dB_len], dtype=int)
totBit = np.zeros([dB_len], dtype=int)
cer = np.empty(shape=(0, 0))
ber = np.empty(shape=(0, 0))
for i in range(dB_len):
    print("\n--------\nSimulating EbNo = {} dB".format(dB_range[i]))
    env.set_EbNo_dB(dB_range[i])

    while (CwErr[i] < minCwErr and totCw[i] + 1 <= maxcw):
        obs = env.reset()
        done = (env.syndrome.sum() == 0)

        while not done:
            action = trainer.compute_single_action(obs)
            obs, _, done, _ = env.step(action)
            # env.render()

        BitErrThis = np.sum(env.chat)
        BitErr[i] = BitErr[i] + BitErrThis
        if BitErrThis > 0:
            CwErr[i] = CwErr[i] + 1

        totCw[i] += 1
        totBit[i] += n
    cer = np.append(cer,CwErr[i] / totCw[i])
    ber = np.append(ber,BitErr[i] / totBit[i])
    print("CwErr:", CwErr[i])
    print("BitErr:", BitErr[i])
    print("TotCw:", totCw[i])
    print("CER:", CwErr[i] / totCw[i])
    print("BER:", BitErr[i] / totBit[i])

if SAVE:
    resdict = {
        "dB_range": dB_range,
        "CwErr": CwErr,
        "BitErr": BitErr,
        "TotCw": totCw,
        "TotBit": totBit,
        "ber": ber,
        "cer": cer
    }

    print("\n****\nSaving files to:\n.mat -->" + save_path + "\n.txt -->" + save_path_txt)
    sio.savemat(save_path, resdict)
    with open(save_path_txt, 'w') as file_txt:
        file_txt.write(str(resdict))

ray.shutdown()

print("done!")

# }}}
# ============================================================== #
