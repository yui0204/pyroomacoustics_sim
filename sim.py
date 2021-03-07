#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for computing the RIR between several sources and receivers in GPU.
@author: yhfu@npu-aslp.org

pyrirgen: https://github.com/phecda-xu/RIR-Generator

"""

import numpy as np
import soundfile as sf
import math
import pyrirgen
import matplotlib.pyplot as plt 
import os
import pyroomacoustics as pra
from IPython.display import display, Audio

c = 340
fs=16000 # Sampling frequency [Hz]

T60 = 0.5    # Time for the RIR to reach 60dB of attenuation [s] 0.2 - 0.8

src_distance = 1.0
src_height = 1.5 # 1.2-1.8

nSamples = 8000 # 0.5s

x = 7
y = 4
z = 3
room_sz = [x, y, z]  # Size of the room [m]

mic_type = "tamago"

e_absorption, max_order = pra.inverse_sabine(T60, room_sz)
m = pra.make_materials(ceiling="ceiling_plasterboard",
                        floor="felt_5mm",
                        east="fibre_absorber_2",
                        west="fibre_absorber_2",
                        north="fibre_absorber_2",
                        south="blinds_half_open",)


mic_middle_point = [x/2, y/2, 1.2] # 1.0-1.5

radius = 0.05
baseangle = 0.25*np.pi
nchannel = 8
pos_rcv = np.c_[[mic_middle_point[0]+radius*np.cos(-4*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*-4), mic_middle_point[2]],
            [mic_middle_point[0]+radius*np.cos(-3*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*-3), mic_middle_point[2]],
            [mic_middle_point[0]+radius*np.cos(-2*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*-2), mic_middle_point[2]],
            [mic_middle_point[0]+radius*np.cos(-1*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*-1), mic_middle_point[2]],
            [mic_middle_point[0]+radius*np.cos(0*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*0), mic_middle_point[2]],
            [mic_middle_point[0]+radius*np.cos(1*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*1), mic_middle_point[2]],
            [mic_middle_point[0]+radius*np.cos(2*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*2), mic_middle_point[2]],
            [mic_middle_point[0]+radius*np.cos(3*baseangle), mic_middle_point[1]+radius*np.sin(baseangle*3), mic_middle_point[2]],]


plt.figure(figsize=(x, y))
plt.scatter(np.array(pos_rcv).T[0], np.array(pos_rcv).T[1], marker=".", c="blue", edgecolors="blue")
plt.scatter(np.array(pos_rcv).T[0][0], np.array(pos_rcv).T[1][0], marker=".", c="red", edgecolors="red")

room_dir = "./rir/"+('%.1f' % x) + '_' + ('%.1f' % y) + '_' + ('%.1f' % z)
if not os.path.exists(room_dir):
    os.mkdir(room_dir)

dir_name = "./rir/"+mic_type + "/" + \
            ('%.1f' % x) + '_' + ('%.1f' % y) + '_' + ('%.1f' % z) + '__' + \
            ('%.1f' % mic_middle_point[0]) + '_' + ('%.1f' % mic_middle_point[1]) + '_' + ('%.1f' % mic_middle_point[2]) + '__' + \
            ('%.1f' % T60) + '_' + ('%.1f' % src_distance)
dir_name = mic_type
if not os.path.exists(os.path.join(room_dir, mic_type)):
    os.mkdir(os.path.join(room_dir, mic_type))

for a in range(-180, 179, 5):
    room = pra.ShoeBox(room_sz, fs=fs, materials=m, max_order=max_order, air_absorption=True)
    room.add_microphone_array(pos_rcv)

    print(x, y, z, mic_type, a)

    pos_src1 = [mic_middle_point[0]+src_distance*np.cos((a+180)/360*2*np.pi), mic_middle_point[1]+src_distance*np.sin((a+180)/360*2*np.pi), src_height]

    if a == 0:
        plt.scatter(pos_src1[0], pos_src1[1], marker=".", c="red", edgecolors="red")
    else:
        plt.scatter(pos_src1[0], pos_src1[1], marker=".", c="blue", edgecolors="blue")

    impulse = np.zeros([nSamples // 10])
    impulse[0] = 1.0
    room.add_source(pos_src1, signal=impulse)
    room.simulate()
    simulation_data = room.mic_array.signals # シミュレーション音源
    #impulse_responses =room.compute_rir()
    #print(simulation_data.shape)
    sf.write(os.path.join(room_dir, mic_type, "pyroomacoustics_"+('%.1f' % a) + '.wav'), simulation_data.T, fs)

plt.xlim(0, x)
plt.ylim(0, y)
plt.savefig(room_dir + "/layout.png")
plt.close()
