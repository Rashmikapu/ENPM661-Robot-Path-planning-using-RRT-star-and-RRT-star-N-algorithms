ENPM 661 Robot Planning Project 5

Team:
Rashmi Kapu       119461754      
Sameer Arjun S    119385876      



GITHUB LINK :


https://github.com/Rashmikapu/ENPM661-Robot-Path-planning-using-RRT-star-and-RRT-star-N-algorithms#enpm661-robot-path-planning-using-rrt-star-and-rrt-star-n-algorithms


Necessary libraries for the program:
import math
import cv2 
import numpy as np
import random
import copy
import time

NOTE:
The project works on RRT* based algorithms with a focus to improve the computational time required for generating an optimal path.
Since, the location of the random node can be anywhere in the obstacle space, the timed testing of the program varies in each iteration by some value.
This error has been rectified by taking the average value of the tests.

Currently, some test cases are preferred in the program which can however be edited for other test cases.
Additionally, to get a better visualisation in each iteration in RRT*N algorithm, a pause has been added to each frame when the path is generated, where in node generation pattern can be observed and shortest path to each node can be visualised, please click on any key to move to the next frame.


