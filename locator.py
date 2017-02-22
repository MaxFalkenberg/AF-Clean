"""
Script which demonstrates the locating of re-entrant circuits using random forest decision trees.
Should produce an animation for Moviepy (hopefully).
"""

import analysis_theano as at
import time
from PIL import Image
import numpy as np
from Functions import ani_convert
import propagate_singlesource as ps
import moviepy.editor as mpy
import copy


# Parameter dictionary.
# t: time counter
# 'process_list': temporary list of ECG's (gets reset when feature extraction is done.)
grid = {'t':0, 'current_frame':None ,'process_list':[]}
# length of time for recording -> process (should be set to cover at least two waveform periods)
process_length = 5

# Initialising the heart tissue (initially pick where the ecptopic beat is.)
a = ps.Heart(nu=0.2, fakedata=True)
x_pos = int(raw_input("Crit x position: "))
y_pos = int(raw_input("Crit y position: "))
a.set_pulse(60, [[y_pos], [x_pos]])
# Initial grid for the animation.
animation_grid = np.zeros((200,200,3), dtype=np.uint8)
background = mpy.ImageClip(Image.fromarray(animation_grid, 'RGB'))
print background
grid['current_frame'] = background.get_frame(0)

# MODEL AND MACHINE LEARNING PROCESSES.

# def ecg_processor


# Update which updates the grid and performs ecg measure and feature gathering if certain conditions are met.
def update(grid):
    data = a.propagate(ecg=True)
    data = ani_convert(data, shape=a.shape, rp=a.rp, animation_grid=animation_grid)
    # saves current frame to dictionary.
    grid['current_frame'] = data
    # Adding data into the process list (needs to be a deep copy due to reference issues.)
    grid['process_list'].append(copy.deepcopy(data))
    # gives points where the ECG is processed.
    grid['t'] += 1
    if grid['t'] % process_length == 0 and grid['t'] != 0:
        print grid['t']
        print np.array(grid['process_list'])
        # ecg_processor.
        del grid['process_list']
        grid['process_list'] = []


# ANIMATION FUNCTIONS.


def grid_to_npimage(grid):
    image = Image.fromarray(grid['current_frame'], 'RBG')
    print image


# in this case t is the movies duration.
def make_frame(t):
    while grid['t'] < t * process_length:
        update(grid)
    grid_to_npimage(grid)

# animation function placeholder (loops over to simulate saving frames. Each frame moves ECG to new point on grid.)
def animation_ph(duration):
    make_frame(duration)

animation_ph(duration=100)
#
# animation = mpy.VideoClip(make_frame, duration=100)
# animation.write_videofile('mov_test.mp4', fps=20)