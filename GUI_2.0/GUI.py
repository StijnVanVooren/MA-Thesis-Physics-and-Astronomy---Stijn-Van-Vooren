import sys
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtGui
from pyarma import *

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.colors as colors

from os.path import dirname, abspath

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

ind30 = [16,3,14,19,2,8,5,9,30,7,17,20,10,18,6,29,26,21,4,28,1,25,15,24,23,22,12,27,11,13] # Indices for Random Tree N30, as the nodes are renamed on the picture.


plt.close('all')

loader = QUiLoader();

app = QtWidgets.QApplication(sys.argv)
window = loader.load("network_triplets.ui", None) # Loads plot settings GUI, but at RUNTIME!!
window2 = loader.load("plot_settings.ui", None) # Loads network node selector GUI, but at RUNTIME!!

data_path = dirname(abspath(__file__))+"/Data/" #gets the relative path of GUI
# All signals and slots here:

filenames = ["RandomTreeN30","CompleteTreeN31","KarateClubN34","CompleteTreeN43","CompleteTreeN49","RandomTreeN60","LatticeN64","CompleteTreeN85","BaN100","CompleteTreeN121","CompleteTreeN127"]

# Plot settings window:

def load(filename): # load dataset
    global N,N_trials,T_relax,T_evolve,J_min,J_max,J_res
    T_relax = 0
    # Initialize parameters:    
    # load file INFO    
    filename_INFO = data_path+filename+"/"
    filename = data_path+filename+"/temp/"
    currentLine = 1
    with open(filename_INFO + "INFO.txt") as textFile:
        for line in textFile:
            if(currentLine==2):
                N = int(line)
            elif(currentLine == 5):
                N_trials = int(line)
            elif(currentLine == 8):
                T_relax += int(line)
            elif(currentLine == 11):
                T_relax += int(line)
            elif(currentLine == 14):
                T_evolve = int(line)
            elif(currentLine == 23):
                J_min = float(line)
            elif(currentLine == 26):
                J_max = float(line)
            elif(currentLine == 29):
                J_res = int(line)
            

            currentLine+=1
    textFile.close()

    # compute J values that were simulated out of J_min, J_max and J_res:
    global dJ
    dJ = (J_max-J_min)/(J_res-1.0);
    global J
    J = np.empty(shape=J_res, dtype=float);
    for i in range(J_res):
        J[i] = J_min + i*dJ;     
     
    #load Chi matrix,
    global Chi
    Chi = np.empty(J_res, dtype=np.float32)  # Create an empty NumPy array

    for i in range(J_res):        
        with open(filename+f"Chi_{i}.txt", 'r') as file:
            value = float(file.read())  # Read the float value from the file
            Chi[i] = value  # Assign the value to the corresponding array element

    #load TE1 matrix

    # Create an empty 3-dimensional matrix TE1
    global TE1
    TE1 = np.empty((N, N, J_res))

    # Loop over the bin files
    for j in range(J_res): 
        mat = pyarma.mat(N,N)    
        mat.load(filename+f"TE1_{j}.bin")
        TE1[:,:,j] = mat

    # Create an empty 4-dimensional matrix TE2
    global TE2
    TE2 = np.empty((N, N, N, J_res))

    # Loop over the bin files
    for j in range(J_res):  
        cube = pyarma.cube(N,N,N)   
        cube.load(filename+f"TE2_{j}.bin")
        TE2[:,:,:,j] =  np.array(cube)

    
    # Create an empty 4-dimensional matrix ST
    global S_T
    S_T = np.empty((N, N, N, J_res))

    # Loop over the bin files
    for j in range(J_res):     
        cube = pyarma.cube(N,N,N)   
        cube.load(filename+f"ST_{j}.bin")
        S_T[:,:,:,j] = np.array(cube)
    
    # Create an empty 4-dimensional matrix RT
    global R_T
    R_T = np.empty((N, N, N, J_res))

    # Loop over the bin files
    for j in range(J_res):     
        cube = pyarma.cube(N,N,N)   
        cube.load(filename+f"RT_{j}.bin")
        R_T[:,:,:,j] = np.array(cube)
"""
def load(filename): # load dataset
    global N,N_trials,T_relax,T_evolve,J_min,J_max,J_res
    T_relax = 0
    # Initialize parameters:    
    # load file INFO    
    filename = data_path+filename+"/"
    currentLine = 1
    with open(filename + "INFO.txt") as textFile:
        for line in textFile:
            if(currentLine==2):
                N = int(line)
            elif(currentLine == 5):
                N_trials = int(line)
            elif(currentLine == 8):
                T_relax += int(line)
            elif(currentLine == 11):
                T_relax += int(line)
            elif(currentLine == 14):
                T_evolve = int(line)
            elif(currentLine == 23):
                J_min = float(line)
            elif(currentLine == 26):
                J_max = float(line)
            elif(currentLine == 29):
                J_res = int(line)
            

            currentLine+=1
    textFile.close()

    # compute J values that were simulated out of J_min, J_max and J_res:
    global dJ
    dJ = (J_max-J_min)/(J_res-1.0);
    global J
    J = np.empty(shape=J_res, dtype=float);
    for i in range(J_res):
        J[i] = J_min + i*dJ;     
     
    #load Chi matrix,
    global Chi
    Chi = np.fromfile(filename+'Chi.txt', dtype=float, count=J_res, sep=' ')

    #load TE1 matrix

    # Create an empty 3-dimensional matrix TE1
    global TE1
    TE1 = np.empty((N, N, J_res), dtype=float)

    
    with open(filename+'TE1.txt', 'r') as file:
        for line_num, line in enumerate(file):
            if line_num % 2 == 0:  # Even lines contain d and t values
                d, t = map(int, line.split())
            else:  # Odd lines contain j values
                j_values = list(map(float, line.split()))
                TE1[d, t, :] = j_values

    # Create an empty 4-dimensional matrix TE2
    global TE2
    TE2 = np.empty((N, N, N, J_res),dtype = float)

    with open(filename+'TE2.txt', 'r') as file:
        for line_num, line in enumerate(file):
            if line_num % 2 == 0:  # Even lines contain d and t values
                d1, d2, t = map(int, line.split())
            else:  # Odd lines contain j values
                j_values = list(map(float, line.split()))
                TE2[d1, d2, t, :] = j_values
    
    # Create an empty 4-dimensional matrix ST
    global S_T
    S_T = np.empty((N, N, N, J_res),dtype = float)

    with open(filename+'ST.txt', 'r') as file:
        for line_num, line in enumerate(file):
            if line_num % 2 == 0:  # Even lines contain d and t values
                d1, d2, t = map(int, line.split())
            else:  # Odd lines contain j values
                j_values = list(map(float, line.split()))
                S_T[d1, d2, t, :] = j_values
    
    # Create an empty 4-dimensional matrix RT
    global R_T
    R_T = np.empty((N, N, N, J_res),dtype = float)

    with open(filename+'RT.txt', 'r') as file:
        for line_num, line in enumerate(file):
            if line_num % 2 == 0:  # Even lines contain d and t values
                d1, d2, t = map(int, line.split())
            else:  # Odd lines contain j values
                j_values = list(map(float, line.split()))
                R_T[d1, d2, t, :] = j_values
"""


average_over_d1 = False
average_over_d2 = False
average_over_t = False

def flip_average_over_d1(new_bool):
    global average_over_d1
    average_over_d1 = new_bool  
    #print_bool_state()
window2.checkBox.toggled.connect(flip_average_over_d1)

def flip_average_over_d2(new_bool):
    global average_over_d2
    average_over_d2 = new_bool  
    #print_bool_state()
window2.checkBox_2.toggled.connect(flip_average_over_d2)

def flip_average_over_t(new_bool):
    global average_over_t
    average_over_t = new_bool   
    #print_bool_state()
window2.checkBox_3.toggled.connect(flip_average_over_t)



plot_type_data = True
def set_plot_type_1(boool):
    global plot_type_data
    plot_type_data = boool
window2.checkBox_9.toggled.connect(set_plot_type_1)

plot_type_fit = False
def set_plot_type_2(boool):
    global plot_type_fit
    plot_type_fit = boool
window2.checkBox_10.toggled.connect(set_plot_type_2)

plot_type_peak = False
def set_plot_type_3(boool):
    global plot_type_peak
    plot_type_peak = boool   
window2.checkBox_12.toggled.connect(set_plot_type_3)

plot_type_Chi = True
def set_plot_type_4(boool):
    global plot_type_Chi
    plot_type_Chi = boool   
window2.checkBox_13.toggled.connect(set_plot_type_4)

plot_type_Chi_line = True
def set_plot_type_5(boool):
    global plot_type_Chi_line
    plot_type_Chi_line = boool   
window2.checkBox_14.toggled.connect(set_plot_type_5)

data_type_te1 = False
def set_data_type_1(boool):
    global data_type_te1
    data_type_te1 = boool
window2.checkBox_17.toggled.connect(set_data_type_1)

data_type_te2 = False
def set_data_type_2(boool):
    global data_type_te2
    data_type_te2 = boool
window2.checkBox_18.toggled.connect(set_data_type_2)

data_type_te12 = False
def set_data_type_3(boool):
    global data_type_te12
    data_type_te12 = boool
window2.checkBox_19.toggled.connect(set_data_type_3)

data_type_st = True
def set_data_type_4(boool):
    global data_type_st
    data_type_st = boool
window2.checkBox_21.toggled.connect(set_data_type_4)

data_type_rt = False
def set_data_type_5(boool):
    global data_type_rt
    data_type_rt = boool
window2.checkBox_20.toggled.connect(set_data_type_5)

# Plots:

def plot_type_1_1(Y,data_string,str_d,str_t,normalise):  
    plt.xlabel(r"$\beta$", fontsize=27)  
    if(normalise):
        maxy = np.max(Y)
        if maxy != 0:
            Y = Y/maxy

    plt.scatter(J,Y,label = data_string+"("+str_d+r"$\rightarrow$"+str_t+")",marker = ".") 

def plot_type_1_2(Y, data_string, str_d1, str_d2, str_t,normalise):      
    plt.xlabel(r"$\beta$", fontsize=27)
    marker_str = "o"
    if(data_string == "S_T"):
        marker_str = "+"
    if(data_string == "R_T"):
        marker_str = "_"
    if(normalise):
        maxy = np.max(Y)
        if maxy != 0:
            Y = Y/maxy
    plt.scatter(J,Y,label = data_string+"("+str_d1+","+str_d2+r"$\rightarrow$"+str_t+")", marker = marker_str)

def plot_type_2_1(Y,data_string,str_d,str_t,normalise):
    plt.xlabel(r"$\beta$", fontsize=27)
    #Fit Y:
    # Create polynomial features of degree 20
    poly = PolynomialFeatures(degree=window2.spinBox.value())
    X_poly = poly.fit_transform(J.reshape(-1, 1))
    # Fit the polynomial regression model
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, Y)
    # Predict the response for a sequence of X values
    X_seq = np.linspace(J_min, J_max, num=400).reshape(-1, 1)
    X_seq_poly = poly.fit_transform(X_seq)
    Y_seq = poly_reg.predict(X_seq_poly)
    #Plot the original data and the predicted values   
    if(normalise):
        maxy = np.max(Y_seq)
        if maxy != 0:
            Y_seq = Y_seq/maxy
    plt.plot(X_seq, Y_seq,label = data_string+"("+str_d+r"$\rightarrow$"+str_t+")"+" fit")

def plot_type_2_2(Y, data_string, str_d1, str_d2, str_t,normalise):
    plt.xlabel(r"$\beta$", fontsize=27)
    #Fit Y:
    # Create polynomial features of degree 20
    poly = PolynomialFeatures(degree=window2.spinBox.value())
    X_poly = poly.fit_transform(J.reshape(-1, 1))
    # Fit the polynomial regression model
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, Y)
    # Predict the response for a sequence of X values
    X_seq = np.linspace(J_min, J_max, num=400).reshape(-1, 1)
    X_seq_poly = poly.fit_transform(X_seq)
    Y_seq = poly_reg.predict(X_seq_poly)
    if(normalise):
        maxy = np.max(Y_seq)
        if maxy != 0:
            Y_seq = Y_seq/maxy
    #Plot the original data and the predicted values    
    plt.plot(X_seq, Y_seq,label = data_string+"("+str_d1+","+str_d2+r"$\rightarrow$"+str_t+")"+" fit")

def plot_type_3_1(Y,data_string,str_d,str_t,normalise):
    plt.xlabel(r"$\beta$", fontsize=27)
    # create polynomial features up to degree 20
    poly = PolynomialFeatures(degree=window2.spinBox.value())

    # fit polynomial features to data
    X_poly = poly.fit_transform(J.reshape(-1, 1))

    # fit linear regression model to polynomial features
    model = LinearRegression()
    model.fit(X_poly, Y)

    # Predict the response for a sequence of X values
    X_grid = np.linspace(J_min, J_max, num=400).reshape(-1, 1)
    X_grid_poly = poly.transform(X_grid)
    Y_pred = model.predict(X_grid_poly)

    # find maximum point of Y_pred
    max_idx = np.argmax(Y_pred)
    max_point = (X_grid[max_idx], Y_pred[max_idx])
    if(normalise):
        max_point = max_point[0], 1

    # plot the data, the fitted curve, and the maximum point
    #plt.plot(J, Y, 'ro')
    #plt.plot(X_grid, Y_pred, 'b-')
    plt.plot(*max_point,'x' ,label = "max "+data_string+"("+str_d+r"$\rightarrow$"+str_t+")")

def plot_type_3_2(Y, data_string, str_d1, str_d2, str_t,normalise):
    plt.xlabel(r"$\beta$", fontsize=27)
    # create polynomial features up to degree 20
    poly = PolynomialFeatures(degree=window2.spinBox.value())

    # fit polynomial features to data
    X_poly = poly.fit_transform(J.reshape(-1, 1))

    # fit linear regression model to polynomial features
    model = LinearRegression()
    model.fit(X_poly, Y)

    # Predict the response for a sequence of X values
    X_grid = np.linspace(J_min, J_max, num=400).reshape(-1, 1)
    X_grid_poly = poly.transform(X_grid)
    Y_pred = model.predict(X_grid_poly)

    # find maximum point of Y_pred
    max_idx = np.argmax(Y_pred)
    max_point = (X_grid[max_idx], Y_pred[max_idx])
    if(normalise):
        max_point = max_point[0], 1

    # plot the data, the fitted curve, and the maximum point
    #plt.plot(J, Y, 'ro')
    #plt.plot(X_grid, Y_pred, 'b-')
    plt.plot(*max_point,'x' ,label = "max "+data_string+"("+str_d1+","+str_d2+r"$\rightarrow$"+str_t+")")

def plot_all_types_1(Y,data_string,str_d,str_t,normalise):
    if(plot_type_data):
        plot_type_1_1(Y,data_string,str_d,str_t,normalise)
    if(plot_type_fit):
        plot_type_2_1(Y,data_string,str_d,str_t,normalise)
    if(plot_type_peak):
        plot_type_3_1(Y,data_string,str_d,str_t,normalise)

def plot_all_types_2(Y, data_string, str_d1, str_d2, str_t,normalise):
    if(plot_type_data):
        plot_type_1_2(Y, data_string, str_d1, str_d2, str_t,normalise)
    if(plot_type_fit):
        plot_type_2_2(Y, data_string, str_d1, str_d2, str_t,normalise)
    if(plot_type_peak):
        plot_type_3_2(Y, data_string, str_d1, str_d2, str_t,normalise)

global fig
global ax
fig, ax = plt.subplots()
plt.ion()  # Turn on interactive mode

max_data = 0
load(filenames[1])
max_chi = np.max(Chi)

def plot(normalise):
    global fig
    global ax 
    global max_data

    # Set the default colormap to 'magma'
    plt.set_cmap('magma')

    plt.rcParams['image.cmap'] = 'magma'

    #Reopen figure if it is closed
    if(plt.get_fignums()):
        plt.show(block=False)
        # Draw horizontal line
        #plt.axhline(0)
    max_te1 = 0
    max_te2 = 0
    max_te12 = 0
    max_st = 0
    max_rt = 0

    d1_len = len(d1)
    d2_len = len(d2)
    t_len = len(t)

    d1_range = range(d1_len)
    d2_range = range(d2_len)
    t_range = range(t_len)

    if(data_type_te1 == True):
        Y_te1 = TE1[d1][:,t]
        axes = ()
        if average_over_d1:
            axes += (0,)
        if average_over_t:
            axes += (1,)
        Y_te1 = np.mean(Y_te1, axis = axes)
        max_te1 = np.max(Y_te1)

        if not average_over_d1 and not average_over_t:
            for d1_ind in d1_range:
                for t_ind in t_range:
                    plot_all_types_1(Y_te1[d1_ind,t_ind,:],"TE",str(d1[d1_ind]+1),str(t[t_ind]+1),normalise)
        if average_over_d1 and not average_over_t:
            for t_ind in t_range:
                    plot_all_types_1(Y_te1[t_ind,:],"TE","AVG",str(t[t_ind]+1),normalise)
        if not average_over_d1 and average_over_t:
            for d1_ind in d1_range:
                plot_all_types_1(Y_te1[d1_ind,:],"TE",str(d1[d1_ind]+1),"AVG",normalise)
        if average_over_d1 and average_over_t:
            plot_all_types_1(Y_te1,"TE","AVG","AVG",normalise)


    if(data_type_te2 == True):
        Y_te2 = TE1[d2][:,t]
        axes = ()
        if average_over_d2:
            axes += (0,)
        if average_over_t:
            axes += (1,)
        Y_te2 = np.mean(Y_te2, axis = axes)
        max_te2 = np.max(Y_te2)

        if not average_over_d2 and not average_over_t:
            for d2_ind in d2_range:
                for t_ind in t_range:
                    plot_all_types_1(Y_te2[d2_ind,t_ind,:],"TE",str(d2[d2_ind]+1),str(t[t_ind]+1),normalise)
        if average_over_d2 and not average_over_t:
            for t_ind in t_range:
                    plot_all_types_1(Y_te2[t_ind,:],"TE","AVG",str(t[t_ind]+1),normalise)
        if not average_over_d2 and average_over_t:
            for d2_ind in d2_range:
                plot_all_types_1(Y_te2[d2_ind,:],"TE",str(d2[d2_ind]+1),"AVG",normalise)
        if average_over_d2 and average_over_t:
            plot_all_types_1(Y_te2,"TE","AVG","AVG",normalise)

    if(data_type_te12 == True):
        Y_te12 = TE2[t][:,d2][:,:,d1]
        axes = ()
        if average_over_d1:
            axes += (2,)
        if average_over_d2:
            axes += (1,)
        if average_over_t:
            axes += (0,)
        Y_te12 = np.mean(Y_te12, axis = axes)
        max_te12 = np.max(Y_te12)

        if not average_over_d1 and not average_over_d2 and not average_over_t:
            for d1_ind in d1_range:
                for d2_ind in d2_range:
                    for t_ind in t_range:
                        plot_all_types_2(Y_te12[t_ind,d2_ind,d1_ind,:], "TE", str(d1[d1_ind]+1), str(d2[d2_ind]+1), str(t[t_ind]+1),normalise)

        if average_over_d1 and not average_over_d2 and not average_over_t:
            for d2_ind in d2_range:
                for t_ind in t_range:
                    plot_all_types_2(Y_te12[t_ind,d2_ind,:], "TE", "AVG", str(d2[d2_ind]+1), str(t[t_ind]+1),normalise)
        
        if not average_over_d1 and average_over_d2 and not average_over_t:
            for d1_ind in d1_range:
                for t_ind in t_range:
                    plot_all_types_2(Y_te12[t_ind,d1_ind,:], "TE", str(d1[d1_ind]+1), "AVG", str(t[t_ind]+1),normalise)
        
        if not average_over_d1 and not average_over_d2 and average_over_t:
            for d1_ind in d1_range:
                for d2_ind in d2_range:
                    plot_all_types_2(Y_te12[d2_ind,d1_ind,:], "TE", str(d1[d1_ind]+1), str(d2[d2_ind]+1), "AVG",normalise)
        
        if average_over_d1 and average_over_d2 and not average_over_t:
            for t_ind in t_range:
                plot_all_types_2(Y_te12[t_ind,:], "TE", "AVG", "AVG", str(t[t_ind]+1),normalise)

        if average_over_d1 and not average_over_d2 and average_over_t:
            for d2_ind in d2_range:
                plot_all_types_2(Y_te12[d2_ind,:], "TE", "AVG", str(d2[d2_ind]+1), "AVG",normalise)
        if not average_over_d1 and average_over_d2 and average_over_t:
            for d1_ind in d1_range:
                plot_all_types_2(Y_te12[d1_ind,:], "TE", str(d1[d1_ind]+1), "AVG", "AVG",normalise)

        if average_over_d1 and average_over_d2 and average_over_t:
            plot_all_types_2(Y_te12, "TE", "AVG", "AVG", "AVG",normalise)


    if(data_type_st == True):
        Y_st = S_T[t][:,d2][:,:,d1]
        axes = ()
        if average_over_d1:
            axes += (2,)
        if average_over_d2:
            axes += (1,)
        if average_over_t:
            axes += (0,)
        Y_st = np.mean(Y_st, axis = axes)
        max_st = np.max(Y_st)

        if not average_over_d1 and not average_over_d2 and not average_over_t:
            for d1_ind in d1_range:
                for d2_ind in d2_range:
                    for t_ind in t_range:
                        plot_all_types_2(Y_st[t_ind,d2_ind,d1_ind,:], "S_T", str(d1[d1_ind]+1), str(d2[d2_ind]+1), str(t[t_ind]+1),normalise)

        if average_over_d1 and not average_over_d2 and not average_over_t:
            for d2_ind in d2_range:
                for t_ind in t_range:
                    plot_all_types_2(Y_st[t_ind,d2_ind,:], "S_T", "AVG", str(d2[d2_ind]+1), str(t[t_ind]+1),normalise)
        
        if not average_over_d1 and average_over_d2 and not average_over_t:
            for d1_ind in d1_range:
                for t_ind in t_range:
                    plot_all_types_2(Y_st[t_ind,d1_ind,:], "S_T", str(d1[d1_ind]+1), "AVG", str(t[t_ind]+1),normalise)
        
        if not average_over_d1 and not average_over_d2 and average_over_t:
            for d1_ind in d1_range:
                for d2_ind in d2_range:
                    plot_all_types_2(Y_st[d2_ind,d1_ind,:], "S_T", str(d1[d1_ind]+1), str(d2[d2_ind]+1), "AVG",normalise)
        
        if average_over_d1 and average_over_d2 and not average_over_t:
            for t_ind in t_range:
                plot_all_types_2(Y_st[t_ind,:], "S_T", "AVG", "AVG", str(t[t_ind]+1),normalise)

        if average_over_d1 and not average_over_d2 and average_over_t:
            for d2_ind in d2_range:
                plot_all_types_2(Y_st[d2_ind,:], "S_T", "AVG", str(d2[d2_ind]+1), "AVG",normalise)
        if not average_over_d1 and average_over_d2 and average_over_t:
            for d1_ind in d1_range:
                plot_all_types_2(Y_st[d1_ind,:], "S_T", str(d1[d1_ind]+1), "AVG", "AVG",normalise)

        if average_over_d1 and average_over_d2 and average_over_t:
            plot_all_types_2(Y_st, "S_T", "AVG", "AVG", "AVG",normalise)

    if(data_type_rt == True):
        Y_rt = R_T[t][:,d2][:,:,d1]
        axes = ()
        if average_over_d1:
            axes += (2,)
        if average_over_d2:
            axes += (1,)
        if average_over_t:
            axes += (0,)
        Y_rt = np.mean(Y_rt, axis = axes)
        max_rt = np.max(Y_rt)

        if not average_over_d1 and not average_over_d2 and not average_over_t:
            for d1_ind in d1_range:
                for d2_ind in d2_range:
                    for t_ind in t_range:
                        plot_all_types_2(Y_rt[t_ind,d2_ind,d1_ind,:], "R_T", str(d1[d1_ind]+1), str(d2[d2_ind]+1), str(t[t_ind]+1),normalise)

        if average_over_d1 and not average_over_d2 and not average_over_t:
            for d2_ind in d2_range:
                for t_ind in t_range:
                    plot_all_types_2(Y_rt[t_ind,d2_ind,:], "R_T", "AVG", str(d2[d2_ind]+1), str(t[t_ind]+1),normalise)
        
        if not average_over_d1 and average_over_d2 and not average_over_t:
            for d1_ind in d1_range:
                for t_ind in t_range:
                    plot_all_types_2(Y_rt[t_ind,d1_ind,:], "R_T", str(d1[d1_ind]+1), "AVG", str(t[t_ind]+1),normalise)
        
        if not average_over_d1 and not average_over_d2 and average_over_t:
            for d1_ind in d1_range:
                for d2_ind in d2_range:
                    plot_all_types_2(Y_rt[d2_ind,d1_ind,:], "R_T", str(d1[d1_ind]+1), str(d2[d2_ind]+1), "AVG",normalise)
        
        if average_over_d1 and average_over_d2 and not average_over_t:
            for t_ind in t_range:
                plot_all_types_2(Y_rt[t_ind,:], "R_T", "AVG", "AVG", str(t[t_ind]+1),normalise)

        if average_over_d1 and not average_over_d2 and average_over_t:
            for d2_ind in d2_range:
                plot_all_types_2(Y_rt[d2_ind,:], "R_T", "AVG", str(d2[d2_ind]+1), "AVG",normalise)
        if not average_over_d1 and average_over_d2 and average_over_t:
            for d1_ind in d1_range:
                plot_all_types_2(Y_rt[d1_ind,:], "R_T", str(d1[d1_ind]+1), "AVG", "AVG",normalise)

        if average_over_d1 and average_over_d2 and average_over_t:
            plot_all_types_2(Y_rt, "R_T", "AVG", "AVG", "AVG",normalise)
    

    if plot_type_Chi:
        if not normalise:
            max_data = np.max([max_te1,max_te2,max_te12,max_st,max_rt,max_data])
            plt.ylim(0,1.025*max_data) 
            scaling = max_data/max_chi
            plt.plot(J,Chi*scaling,label = r"$\chi$")
        else:
            plt.plot(J,Chi/np.max(Chi),label = r"$\chi$")
            plt.ylim(0,1.025)

    if plot_type_Chi_line:
        max_index = np.argmax(Chi)
        max_J = J[max_index]
        plt.axvline(x=max_J, color='r', linestyle='--')

    plt.ylabel("bits", fontsize=27)
    plt.legend(fontsize = 14)
    plt.title(window2.textEdit.toPlainText(), fontsize=27)
    fig.canvas.draw()

window2.pushButton_33.clicked.connect(lambda: plot(False))
window2.pushButton_34.clicked.connect(lambda: plot(True))


def clear_plot():    
    global fig
    global ax
    global max_data
    max_data = 0
    fig.clear()
    ax.clear()
    # Draw horizontal line
    #plt.axhline(0)
    fig.canvas.draw()
window2.pushButton_35.clicked.connect(clear_plot)



# Network selector window:

d1 = [];
d2 = [];
t = [];

setting_d1 = True
setting_d2 = False
setting_t = False


def setting_d1_fun():
    global setting_d1,setting_d2,setting_t
    setting_d1 = True
    setting_d2 = False
    setting_t = False

def setting_d2_fun():
    global setting_d1,setting_d2,setting_t
    setting_d1 = False
    setting_d2 = True
    setting_t = False

def setting_t_fun():
    global setting_d1,setting_d2,setting_t
    setting_d1 = False
    setting_d2 = False
    setting_t = True



def update_text_triplets_d1(plain_text_edit):
    text = ' | '.join(str(i+1) for i in d1)
    plain_text_edit.setPlainText(text)

def update_text_triplets_d2(plain_text_edit):
    text = ' | '.join(str(i+1) for i in d2)
    plain_text_edit.setPlainText(text)

def update_text_triplets_t(plain_text_edit):
    text = ' | '.join(str(i+1) for i in t)
    plain_text_edit.setPlainText(text)

def parse_text_triplets_d1(plain_text_edit):
    text = plain_text_edit.toPlainText()
    global d1
    d1 = []
    try:
        integers = text.split('|')
        for integer in integers:
            stripped_integer = integer.strip()
            if stripped_integer:
                d1.append(int(stripped_integer))
    except ValueError:
        print("Invalid format.")

def parse_text_triplets_d2(plain_text_edit):
    text = plain_text_edit.toPlainText()
    global d2
    d2 = []
    try:
        integers = text.split('|')
        for integer in integers:
            stripped_integer = integer.strip()
            if stripped_integer:
                d2.append(int(stripped_integer))
    except ValueError:
        print("Invalid format.")

def parse_text_triplets_t(plain_text_edit):
    text = plain_text_edit.toPlainText()
    global t
    t = []
    try:
        integers = text.split('|')
        for integer in integers:
            stripped_integer = integer.strip()
            if stripped_integer:
                t.append(int(stripped_integer))
    except ValueError:
        print("Invalid format.")


def parse_plain_texts_d1(current_tab):
    if current_tab == 1:
        parse_text_triplets_d1(window.plainTextEdit_4)
    elif current_tab == 2:
        parse_text_triplets_d1(window.plainTextEdit_10)
    elif current_tab == 3:
        parse_text_triplets_d1(window.plainTextEdit_13)
    elif current_tab == 4:
        parse_text_triplets_d1(window.plainTextEdit_16)
    elif current_tab == 5:
        parse_text_triplets_d1(window.plainTextEdit_19)
    elif current_tab == 6:
        parse_text_triplets_d1(window.plainTextEdit_7)
    elif current_tab == 7:
        parse_text_triplets_d1(window.plainTextEdit_28)
    elif current_tab == 8:
        parse_text_triplets_d1(window.plainTextEdit_22)
    elif current_tab == 9:
        parse_text_triplets_d1(window.plainTextEdit)
    elif current_tab == 10:
        parse_text_triplets_d1(window.plainTextEdit_25)
    elif current_tab == 11:
        parse_text_triplets_d1(window.plainTextEdit_31)

def parse_plain_texts_d2(current_tab):   
    if current_tab == 1:
        parse_text_triplets_d2(window.plainTextEdit_5)
    elif current_tab == 2:
        parse_text_triplets_d2(window.plainTextEdit_11)
    elif current_tab == 3:
        parse_text_triplets_d2(window.plainTextEdit_14)
    elif current_tab == 4:
        parse_text_triplets_d2(window.plainTextEdit_17)
    elif current_tab == 5:
        parse_text_triplets_d2(window.plainTextEdit_20)
    elif current_tab == 6:
        parse_text_triplets_d2(window.plainTextEdit_8)
    elif current_tab == 7:
        parse_text_triplets_d2(window.plainTextEdit_29)
    elif current_tab == 8:
        parse_text_triplets_d2(window.plainTextEdit_23)
    elif current_tab == 9:
        parse_text_triplets_d2(window.plainTextEdit_2)
    elif current_tab == 10:
        parse_text_triplets_d2(window.plainTextEdit_26)
    elif current_tab == 11:
        parse_text_triplets_d2(window.plainTextEdit_32)

def parse_plain_texts_t(current_tab):
    if current_tab == 1:
        parse_text_triplets_t(window.plainTextEdit_6)
    elif current_tab == 2:
        parse_text_triplets_t(window.plainTextEdit_12)
    elif current_tab == 3:
        parse_text_triplets_t(window.plainTextEdit_15)
    elif current_tab == 4:
        parse_text_triplets_t(window.plainTextEdit_18)
    elif current_tab == 5:
        parse_text_triplets_t(window.plainTextEdit_21)
    elif current_tab == 6:
        parse_text_triplets_t(window.plainTextEdit_9)
    elif current_tab == 7:
        parse_text_triplets_t(window.plainTextEdit_30)
    elif current_tab == 8:
        parse_text_triplets_t(window.plainTextEdit_24)
    elif current_tab == 9:
        parse_text_triplets_t(window.plainTextEdit_3)
    elif current_tab == 10:
        parse_text_triplets_t(window.plainTextEdit_27)
    elif current_tab == 11:
        parse_text_triplets_t(window.plainTextEdit_33)

def parse_plain_texts(current_tab):
    parse_plain_texts_d1(current_tab)
    parse_plain_texts_d2(current_tab)
    parse_plain_texts_t(current_tab)

def update_plain_texts_d1(current_tab):
    if current_tab == 1:
        update_text_triplets_d1(window.plainTextEdit_4)
    elif current_tab == 2:
        update_text_triplets_d1(window.plainTextEdit_10)
    elif current_tab == 3:
        update_text_triplets_d1(window.plainTextEdit_13)
    elif current_tab == 4:
        update_text_triplets_d1(window.plainTextEdit_16)
    elif current_tab == 5:
        update_text_triplets_d1(window.plainTextEdit_19)
    elif current_tab == 6:
        update_text_triplets_d1(window.plainTextEdit_7)
    elif current_tab == 7:
        update_text_triplets_d1(window.plainTextEdit_28)
    elif current_tab == 8:
        update_text_triplets_d1(window.plainTextEdit_22)
    elif current_tab == 9:
        update_text_triplets_d1(window.plainTextEdit)
    elif current_tab == 10:
        update_text_triplets_d1(window.plainTextEdit_25)
    elif current_tab == 11:
        update_text_triplets_d1(window.plainTextEdit_31)

def update_plain_texts_d2(current_tab):   
    if current_tab == 1:
        update_text_triplets_d2(window.plainTextEdit_5)
    elif current_tab == 2:
        update_text_triplets_d2(window.plainTextEdit_11)
    elif current_tab == 3:
        update_text_triplets_d2(window.plainTextEdit_14)
    elif current_tab == 4:
        update_text_triplets_d2(window.plainTextEdit_17)
    elif current_tab == 5:
        update_text_triplets_d2(window.plainTextEdit_20)
    elif current_tab == 6:
        update_text_triplets_d2(window.plainTextEdit_8)
    elif current_tab == 7:
        update_text_triplets_d2(window.plainTextEdit_29)
    elif current_tab == 8:
        update_text_triplets_d2(window.plainTextEdit_23)
    elif current_tab == 9:
        update_text_triplets_d2(window.plainTextEdit_2)
    elif current_tab == 10:
        update_text_triplets_d2(window.plainTextEdit_26)
    elif current_tab == 11:
        update_text_triplets_d2(window.plainTextEdit_32)

def update_plain_texts_t(current_tab):
    if current_tab == 1:
        update_text_triplets_t(window.plainTextEdit_6)
    elif current_tab == 2:
        update_text_triplets_t(window.plainTextEdit_12)
    elif current_tab == 3:
        update_text_triplets_t(window.plainTextEdit_15)
    elif current_tab == 4:
        update_text_triplets_t(window.plainTextEdit_18)
    elif current_tab == 5:
        update_text_triplets_t(window.plainTextEdit_21)
    elif current_tab == 6:
        update_text_triplets_t(window.plainTextEdit_9)
    elif current_tab == 7:
        update_text_triplets_t(window.plainTextEdit_30)
    elif current_tab == 8:
        update_text_triplets_t(window.plainTextEdit_24)
    elif current_tab == 9:
        update_text_triplets_t(window.plainTextEdit_3)
    elif current_tab == 10:
        update_text_triplets_t(window.plainTextEdit_27)
    elif current_tab == 11:
        update_text_triplets_t(window.plainTextEdit_33)

def update_plain_texts(current_tab):
    update_plain_texts_d1(current_tab)
    update_plain_texts_d2(current_tab)
    update_plain_texts_t(current_tab)

def add_new_node_to_triplet(node):    
    current_tab = window.currentIndex() +1
    global d1,d2,t
    if(setting_d1):
        if node not in d1:
            d1.append(node)
            update_plain_texts_d1(window.currentIndex()+1)
    if(setting_d2):
        if node not in d2:
            d2.append(node)
            
            update_plain_texts_d2(window.currentIndex()+1)
    if(setting_t):
        if node not in t:
            t.append(node)
            
            update_plain_texts_t(window.currentIndex()+1)
    




window.pushButton_34.clicked.connect(setting_d1_fun)
window.pushButton_35.clicked.connect(setting_d2_fun)
window.pushButton_36.clicked.connect(setting_t_fun)
window.pushButton_40.clicked.connect(setting_d1_fun)
window.pushButton_41.clicked.connect(setting_d2_fun)
window.pushButton_42.clicked.connect(setting_t_fun)
window.pushButton_43.clicked.connect(setting_d1_fun)
window.pushButton_44.clicked.connect(setting_d2_fun)
window.pushButton_45.clicked.connect(setting_t_fun)
window.pushButton_37.clicked.connect(setting_d1_fun)
window.pushButton_38.clicked.connect(setting_d2_fun)
window.pushButton_39.clicked.connect(setting_t_fun)
window.pushButton_31.clicked.connect(setting_d1_fun)
window.pushButton_32.clicked.connect(setting_d2_fun)
window.pushButton_33.clicked.connect(setting_t_fun)

window.pushButton_302.clicked.connect(setting_d1_fun)
window.pushButton_305.clicked.connect(setting_d2_fun)
window.pushButton_308.clicked.connect(setting_t_fun)
window.pushButton_311.clicked.connect(setting_d1_fun)
window.pushButton_314.clicked.connect(setting_d2_fun)
window.pushButton_317.clicked.connect(setting_t_fun)

window.pushButton_338.clicked.connect(setting_d1_fun)
window.pushButton_341.clicked.connect(setting_d2_fun)
window.pushButton_344.clicked.connect(setting_t_fun)
window.pushButton_320.clicked.connect(setting_d1_fun)
window.pushButton_323.clicked.connect(setting_d2_fun)
window.pushButton_326.clicked.connect(setting_t_fun)

window.pushButton_329.clicked.connect(setting_d1_fun)
window.pushButton_332.clicked.connect(setting_d2_fun)
window.pushButton_335.clicked.connect(setting_t_fun)
window.pushButton_710.clicked.connect(setting_d1_fun)
window.pushButton_713.clicked.connect(setting_d2_fun)
window.pushButton_716.clicked.connect(setting_t_fun)

def set_all_d1():
    global d1
    d1 = list(range(N))
    update_plain_texts_d1(window.currentIndex()+1)

def set_all_d2():
    global d2
    d2 = list(range(N))
    update_plain_texts_d2(window.currentIndex()+1)

def set_all_t():
    global t
    t = list(range(N))
    update_plain_texts_t(window.currentIndex()+1)

window.pushButton_287.clicked.connect(set_all_d1)
window.pushButton_288.clicked.connect(set_all_d2)
window.pushButton_289.clicked.connect(set_all_t)

window.pushButton_290.clicked.connect(set_all_d1)
window.pushButton_291.clicked.connect(set_all_d2)
window.pushButton_292.clicked.connect(set_all_t)

window.pushButton_293.clicked.connect(set_all_d1)
window.pushButton_294.clicked.connect(set_all_d2)
window.pushButton_295.clicked.connect(set_all_t)

window.pushButton_296.clicked.connect(set_all_d1)
window.pushButton_297.clicked.connect(set_all_d2)
window.pushButton_298.clicked.connect(set_all_t)

window.pushButton_303.clicked.connect(set_all_d1)
window.pushButton_306.clicked.connect(set_all_d2)
window.pushButton_309.clicked.connect(set_all_t)

window.pushButton_312.clicked.connect(set_all_d1)
window.pushButton_315.clicked.connect(set_all_d2)
window.pushButton_318.clicked.connect(set_all_t)

window.pushButton_339.clicked.connect(set_all_d1)
window.pushButton_342.clicked.connect(set_all_d2)
window.pushButton_345.clicked.connect(set_all_t)

window.pushButton_321.clicked.connect(set_all_d1)
window.pushButton_324.clicked.connect(set_all_d2)
window.pushButton_327.clicked.connect(set_all_t)

window.pushButton_299.clicked.connect(set_all_d1)
window.pushButton_300.clicked.connect(set_all_d2)
window.pushButton_301.clicked.connect(set_all_t)

window.pushButton_330.clicked.connect(set_all_d1)
window.pushButton_333.clicked.connect(set_all_d2)
window.pushButton_336.clicked.connect(set_all_t)

window.pushButton_711.clicked.connect(set_all_d1)
window.pushButton_714.clicked.connect(set_all_d2)
window.pushButton_717.clicked.connect(set_all_t)

def reset_d1():
    current_tab = window.currentIndex() +1
    global d1
    d1 = []
    update_plain_texts_d1(current_tab)

def reset_d2():
    current_tab = window.currentIndex() +1
    global d2
    d2 = []
    update_plain_texts_d2(current_tab)

def reset_t():
    current_tab = window.currentIndex() +1
    global t
    t = []
    update_plain_texts_t(current_tab)

window.pushButton_272.clicked.connect(reset_d1)
window.pushButton_273.clicked.connect(reset_d2)
window.pushButton_274.clicked.connect(reset_t)
window.pushButton_275.clicked.connect(reset_d1)
window.pushButton_276.clicked.connect(reset_d2)
window.pushButton_277.clicked.connect(reset_t)
window.pushButton_278.clicked.connect(reset_d1)
window.pushButton_279.clicked.connect(reset_d2)
window.pushButton_280.clicked.connect(reset_t)
window.pushButton_281.clicked.connect(reset_d1)
window.pushButton_282.clicked.connect(reset_d2)
window.pushButton_283.clicked.connect(reset_t)
window.pushButton_284.clicked.connect(reset_d1)
window.pushButton_285.clicked.connect(reset_d2)
window.pushButton_286.clicked.connect(reset_t)

window.pushButton_304.clicked.connect(reset_d1)
window.pushButton_307.clicked.connect(reset_d2)
window.pushButton_310.clicked.connect(reset_t)
window.pushButton_313.clicked.connect(reset_d1)
window.pushButton_316.clicked.connect(reset_d2)
window.pushButton_319.clicked.connect(reset_t)

window.pushButton_340.clicked.connect(reset_d1)
window.pushButton_343.clicked.connect(reset_d2)
window.pushButton_346.clicked.connect(reset_t)
window.pushButton_322.clicked.connect(reset_d1)
window.pushButton_325.clicked.connect(reset_d2)
window.pushButton_328.clicked.connect(reset_t)

window.pushButton_331.clicked.connect(reset_d1)
window.pushButton_334.clicked.connect(reset_d2)
window.pushButton_337.clicked.connect(reset_t)
window.pushButton_712.clicked.connect(reset_d1)
window.pushButton_715.clicked.connect(reset_d2)
window.pushButton_718.clicked.connect(reset_t)



print(N)

def handle_tab_change(current_tab_index):
    global max_chi
    current_tab = current_tab_index + 1

    load(filenames[current_tab_index])
    max_chi = np.max(Chi)
    print(TE2.shape)
    print(N)
    parse_plain_texts(current_tab)
    
    

window.currentChanged.connect(handle_tab_change)

#Random tree N=30
window.pushButton.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_2.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_3.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_4.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_5.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_6.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_7.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_8.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_9.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_10.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_11.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_12.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_13.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_14.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_15.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_16.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_17.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_18.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_19.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_20.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_21.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_22.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_23.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_24.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_25.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_26.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_27.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_28.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_29.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_30.clicked.connect(lambda: add_new_node_to_triplet(29))

#Complete tree N=31
window.pushButton_46.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_47.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_48.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_49.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_50.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_51.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_52.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_53.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_55.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_56.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_57.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_58.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_59.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_60.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_61.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_62.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_63.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_64.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_65.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_66.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_67.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_68.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_69.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_70.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_71.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_72.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_73.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_74.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_75.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_76.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_77.clicked.connect(lambda: add_new_node_to_triplet(30))


#Karate club N = 34
window.pushButton_78.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_79.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_80.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_81.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_82.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_83.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_84.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_85.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_86.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_87.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_88.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_89.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_90.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_91.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_92.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_93.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_94.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_95.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_96.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_97.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_98.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_99.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_100.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_101.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_102.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_103.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_104.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_105.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_106.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_107.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_108.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_109.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_110.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_111.clicked.connect(lambda: add_new_node_to_triplet(33))

# Random Tree N = 60
window.pushButton_112.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_113.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_114.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_115.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_116.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_117.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_118.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_119.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_120.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_121.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_122.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_123.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_124.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_125.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_126.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_127.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_128.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_129.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_130.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_131.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_132.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_133.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_134.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_135.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_136.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_137.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_138.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_139.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_140.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_141.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_142.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_143.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_144.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_145.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_146.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_147.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_148.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_149.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_150.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_151.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_152.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_153.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_154.clicked.connect(lambda: add_new_node_to_triplet(42))
window.pushButton_155.clicked.connect(lambda: add_new_node_to_triplet(43))
window.pushButton_156.clicked.connect(lambda: add_new_node_to_triplet(44))
window.pushButton_157.clicked.connect(lambda: add_new_node_to_triplet(45))
window.pushButton_158.clicked.connect(lambda: add_new_node_to_triplet(46))
window.pushButton_159.clicked.connect(lambda: add_new_node_to_triplet(47))
window.pushButton_160.clicked.connect(lambda: add_new_node_to_triplet(48))
window.pushButton_161.clicked.connect(lambda: add_new_node_to_triplet(49))
window.pushButton_162.clicked.connect(lambda: add_new_node_to_triplet(50))
window.pushButton_163.clicked.connect(lambda: add_new_node_to_triplet(51))
window.pushButton_164.clicked.connect(lambda: add_new_node_to_triplet(52))
window.pushButton_165.clicked.connect(lambda: add_new_node_to_triplet(53))
window.pushButton_166.clicked.connect(lambda: add_new_node_to_triplet(54))
window.pushButton_167.clicked.connect(lambda: add_new_node_to_triplet(55))
window.pushButton_168.clicked.connect(lambda: add_new_node_to_triplet(56))
window.pushButton_169.clicked.connect(lambda: add_new_node_to_triplet(57))
window.pushButton_170.clicked.connect(lambda: add_new_node_to_triplet(58))
window.pushButton_171.clicked.connect(lambda: add_new_node_to_triplet(59))


# Scale free N = 100

window.pushButton_172.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_173.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_174.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_175.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_176.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_177.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_178.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_179.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_180.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_181.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_182.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_183.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_184.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_185.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_186.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_187.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_188.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_189.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_190.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_191.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_192.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_193.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_194.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_195.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_196.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_197.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_198.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_199.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_200.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_201.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_202.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_203.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_204.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_205.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_206.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_207.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_208.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_209.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_210.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_211.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_212.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_213.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_214.clicked.connect(lambda: add_new_node_to_triplet(42))
window.pushButton_215.clicked.connect(lambda: add_new_node_to_triplet(43))
window.pushButton_216.clicked.connect(lambda: add_new_node_to_triplet(44))
window.pushButton_217.clicked.connect(lambda: add_new_node_to_triplet(45))
window.pushButton_218.clicked.connect(lambda: add_new_node_to_triplet(46))
window.pushButton_219.clicked.connect(lambda: add_new_node_to_triplet(47))
window.pushButton_220.clicked.connect(lambda: add_new_node_to_triplet(48))
window.pushButton_221.clicked.connect(lambda: add_new_node_to_triplet(49))
window.pushButton_222.clicked.connect(lambda: add_new_node_to_triplet(50))
window.pushButton_223.clicked.connect(lambda: add_new_node_to_triplet(51))
window.pushButton_224.clicked.connect(lambda: add_new_node_to_triplet(52))
window.pushButton_225.clicked.connect(lambda: add_new_node_to_triplet(53))
window.pushButton_226.clicked.connect(lambda: add_new_node_to_triplet(54))
window.pushButton_227.clicked.connect(lambda: add_new_node_to_triplet(55))
window.pushButton_228.clicked.connect(lambda: add_new_node_to_triplet(56))
window.pushButton_229.clicked.connect(lambda: add_new_node_to_triplet(57))
window.pushButton_230.clicked.connect(lambda: add_new_node_to_triplet(58))
window.pushButton_231.clicked.connect(lambda: add_new_node_to_triplet(59))
window.pushButton_232.clicked.connect(lambda: add_new_node_to_triplet(60))
window.pushButton_233.clicked.connect(lambda: add_new_node_to_triplet(61))
window.pushButton_234.clicked.connect(lambda: add_new_node_to_triplet(62))
window.pushButton_235.clicked.connect(lambda: add_new_node_to_triplet(63))
window.pushButton_236.clicked.connect(lambda: add_new_node_to_triplet(64))
window.pushButton_237.clicked.connect(lambda: add_new_node_to_triplet(65))
window.pushButton_238.clicked.connect(lambda: add_new_node_to_triplet(66))
window.pushButton_239.clicked.connect(lambda: add_new_node_to_triplet(67))
window.pushButton_240.clicked.connect(lambda: add_new_node_to_triplet(68))
window.pushButton_241.clicked.connect(lambda: add_new_node_to_triplet(69))
window.pushButton_242.clicked.connect(lambda: add_new_node_to_triplet(70))
window.pushButton_243.clicked.connect(lambda: add_new_node_to_triplet(71))
window.pushButton_244.clicked.connect(lambda: add_new_node_to_triplet(72))
window.pushButton_245.clicked.connect(lambda: add_new_node_to_triplet(73))
window.pushButton_246.clicked.connect(lambda: add_new_node_to_triplet(74))
window.pushButton_247.clicked.connect(lambda: add_new_node_to_triplet(75))
window.pushButton_248.clicked.connect(lambda: add_new_node_to_triplet(76))
window.pushButton_249.clicked.connect(lambda: add_new_node_to_triplet(77))
window.pushButton_250.clicked.connect(lambda: add_new_node_to_triplet(78))
window.pushButton_251.clicked.connect(lambda: add_new_node_to_triplet(79))
window.pushButton_252.clicked.connect(lambda: add_new_node_to_triplet(80))
window.pushButton_253.clicked.connect(lambda: add_new_node_to_triplet(81))
window.pushButton_254.clicked.connect(lambda: add_new_node_to_triplet(82))
window.pushButton_255.clicked.connect(lambda: add_new_node_to_triplet(83))
window.pushButton_256.clicked.connect(lambda: add_new_node_to_triplet(84))
window.pushButton_257.clicked.connect(lambda: add_new_node_to_triplet(85))
window.pushButton_258.clicked.connect(lambda: add_new_node_to_triplet(86))
window.pushButton_259.clicked.connect(lambda: add_new_node_to_triplet(87))
window.pushButton_260.clicked.connect(lambda: add_new_node_to_triplet(88))
window.pushButton_261.clicked.connect(lambda: add_new_node_to_triplet(89))
window.pushButton_262.clicked.connect(lambda: add_new_node_to_triplet(90))
window.pushButton_263.clicked.connect(lambda: add_new_node_to_triplet(91))
window.pushButton_264.clicked.connect(lambda: add_new_node_to_triplet(92))
window.pushButton_265.clicked.connect(lambda: add_new_node_to_triplet(93))
window.pushButton_266.clicked.connect(lambda: add_new_node_to_triplet(94))
window.pushButton_267.clicked.connect(lambda: add_new_node_to_triplet(95))
window.pushButton_268.clicked.connect(lambda: add_new_node_to_triplet(96))
window.pushButton_269.clicked.connect(lambda: add_new_node_to_triplet(97))
window.pushButton_270.clicked.connect(lambda: add_new_node_to_triplet(98))
window.pushButton_271.clicked.connect(lambda: add_new_node_to_triplet(99))

window.pushButton_846.clicked.connect(lambda: load("BaN100"))
window.pushButton_847.clicked.connect(lambda: load("Local1"))
window.pushButton_848.clicked.connect(lambda: load("Local2"))

# Complete tree N=43

window.pushButton_347.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_348.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_349.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_350.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_351.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_352.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_353.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_354.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_355.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_356.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_357.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_358.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_359.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_360.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_361.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_362.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_363.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_364.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_365.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_366.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_367.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_368.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_369.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_370.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_371.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_372.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_373.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_374.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_375.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_376.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_377.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_378.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_379.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_380.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_381.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_382.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_383.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_384.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_385.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_386.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_387.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_388.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_389.clicked.connect(lambda: add_new_node_to_triplet(42))

# Complete tree N=49

window.pushButton_390.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_391.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_392.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_393.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_394.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_395.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_396.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_397.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_398.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_399.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_400.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_401.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_402.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_403.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_404.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_405.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_406.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_407.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_408.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_409.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_410.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_411.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_412.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_413.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_414.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_415.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_416.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_417.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_418.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_419.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_420.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_421.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_422.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_423.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_424.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_425.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_426.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_427.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_428.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_429.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_430.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_431.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_432.clicked.connect(lambda: add_new_node_to_triplet(42))
window.pushButton_433.clicked.connect(lambda: add_new_node_to_triplet(43))
window.pushButton_434.clicked.connect(lambda: add_new_node_to_triplet(44))
window.pushButton_435.clicked.connect(lambda: add_new_node_to_triplet(45))
window.pushButton_436.clicked.connect(lambda: add_new_node_to_triplet(46))
window.pushButton_437.clicked.connect(lambda: add_new_node_to_triplet(47))
window.pushButton_438.clicked.connect(lambda: add_new_node_to_triplet(48))

# Lattice N=64

window.pushButton_439.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_440.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_441.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_442.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_443.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_444.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_445.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_446.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_447.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_448.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_449.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_450.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_451.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_452.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_453.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_454.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_455.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_456.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_457.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_458.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_459.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_460.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_461.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_462.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_463.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_464.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_465.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_466.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_467.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_468.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_469.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_470.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_471.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_472.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_473.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_474.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_475.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_476.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_477.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_478.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_479.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_480.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_481.clicked.connect(lambda: add_new_node_to_triplet(42))
window.pushButton_482.clicked.connect(lambda: add_new_node_to_triplet(43))
window.pushButton_483.clicked.connect(lambda: add_new_node_to_triplet(44))
window.pushButton_484.clicked.connect(lambda: add_new_node_to_triplet(45))
window.pushButton_485.clicked.connect(lambda: add_new_node_to_triplet(46))
window.pushButton_486.clicked.connect(lambda: add_new_node_to_triplet(47))
window.pushButton_487.clicked.connect(lambda: add_new_node_to_triplet(48))
window.pushButton_488.clicked.connect(lambda: add_new_node_to_triplet(49))
window.pushButton_489.clicked.connect(lambda: add_new_node_to_triplet(50))
window.pushButton_490.clicked.connect(lambda: add_new_node_to_triplet(51))
window.pushButton_491.clicked.connect(lambda: add_new_node_to_triplet(52))
window.pushButton_492.clicked.connect(lambda: add_new_node_to_triplet(53))
window.pushButton_493.clicked.connect(lambda: add_new_node_to_triplet(54))
window.pushButton_494.clicked.connect(lambda: add_new_node_to_triplet(55))
window.pushButton_495.clicked.connect(lambda: add_new_node_to_triplet(56))
window.pushButton_496.clicked.connect(lambda: add_new_node_to_triplet(57))
window.pushButton_497.clicked.connect(lambda: add_new_node_to_triplet(58))
window.pushButton_498.clicked.connect(lambda: add_new_node_to_triplet(59))
window.pushButton_499.clicked.connect(lambda: add_new_node_to_triplet(60))
window.pushButton_500.clicked.connect(lambda: add_new_node_to_triplet(61))
window.pushButton_501.clicked.connect(lambda: add_new_node_to_triplet(62))
window.pushButton_502.clicked.connect(lambda: add_new_node_to_triplet(63))

window.pushButton_849.clicked.connect(lambda: load("LatticeN64"))
window.pushButton_850.clicked.connect(lambda: load("Local3"))
window.pushButton_851.clicked.connect(lambda: load("Local4"))
window.pushButton_852.clicked.connect(lambda: load("Local5"))
window.pushButton_853.clicked.connect(lambda: load("Local6"))
window.pushButton_854.clicked.connect(lambda: load("Local7"))
window.pushButton_855.clicked.connect(lambda: load("Local8"))
window.pushButton_856.clicked.connect(lambda: load("Local9"))
window.pushButton_857.clicked.connect(lambda: load("Local10"))
window.pushButton_858.clicked.connect(lambda: load("Local11"))
window.pushButton_859.clicked.connect(lambda: load("Local12"))
window.pushButton_860.clicked.connect(lambda: load("Local13"))
window.pushButton_861.clicked.connect(lambda: load("Local14"))
window.pushButton_862.clicked.connect(lambda: load("Local15"))
window.pushButton_863.clicked.connect(lambda: load("Lattice2"))
# Complete tree N=85
window.pushButton_503.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_504.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_505.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_506.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_507.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_508.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_509.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_510.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_511.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_512.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_513.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_514.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_515.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_516.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_517.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_518.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_519.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_520.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_521.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_522.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_523.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_524.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_525.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_526.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_527.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_528.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_529.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_530.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_531.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_532.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_533.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_534.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_535.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_536.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_537.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_538.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_539.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_540.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_541.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_542.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_543.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_544.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_545.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_546.clicked.connect(lambda: add_new_node_to_triplet(42))
window.pushButton_547.clicked.connect(lambda: add_new_node_to_triplet(43))
window.pushButton_548.clicked.connect(lambda: add_new_node_to_triplet(44))
window.pushButton_549.clicked.connect(lambda: add_new_node_to_triplet(45))
window.pushButton_550.clicked.connect(lambda: add_new_node_to_triplet(46))
window.pushButton_551.clicked.connect(lambda: add_new_node_to_triplet(47))
window.pushButton_552.clicked.connect(lambda: add_new_node_to_triplet(48))
window.pushButton_553.clicked.connect(lambda: add_new_node_to_triplet(49))
window.pushButton_554.clicked.connect(lambda: add_new_node_to_triplet(50))
window.pushButton_555.clicked.connect(lambda: add_new_node_to_triplet(51))
window.pushButton_556.clicked.connect(lambda: add_new_node_to_triplet(52))
window.pushButton_557.clicked.connect(lambda: add_new_node_to_triplet(53))
window.pushButton_558.clicked.connect(lambda: add_new_node_to_triplet(54))
window.pushButton_559.clicked.connect(lambda: add_new_node_to_triplet(55))
window.pushButton_560.clicked.connect(lambda: add_new_node_to_triplet(56))
window.pushButton_561.clicked.connect(lambda: add_new_node_to_triplet(57))
window.pushButton_562.clicked.connect(lambda: add_new_node_to_triplet(58))
window.pushButton_563.clicked.connect(lambda: add_new_node_to_triplet(59))
window.pushButton_564.clicked.connect(lambda: add_new_node_to_triplet(60))
window.pushButton_565.clicked.connect(lambda: add_new_node_to_triplet(61))
window.pushButton_566.clicked.connect(lambda: add_new_node_to_triplet(62))
window.pushButton_567.clicked.connect(lambda: add_new_node_to_triplet(63))
window.pushButton_568.clicked.connect(lambda: add_new_node_to_triplet(64))
window.pushButton_569.clicked.connect(lambda: add_new_node_to_triplet(65))
window.pushButton_570.clicked.connect(lambda: add_new_node_to_triplet(66))
window.pushButton_571.clicked.connect(lambda: add_new_node_to_triplet(67))
window.pushButton_572.clicked.connect(lambda: add_new_node_to_triplet(68))
window.pushButton_573.clicked.connect(lambda: add_new_node_to_triplet(69))
window.pushButton_574.clicked.connect(lambda: add_new_node_to_triplet(70))
window.pushButton_575.clicked.connect(lambda: add_new_node_to_triplet(71))
window.pushButton_576.clicked.connect(lambda: add_new_node_to_triplet(72))
window.pushButton_577.clicked.connect(lambda: add_new_node_to_triplet(73))
window.pushButton_578.clicked.connect(lambda: add_new_node_to_triplet(74))
window.pushButton_579.clicked.connect(lambda: add_new_node_to_triplet(75))
window.pushButton_580.clicked.connect(lambda: add_new_node_to_triplet(76))
window.pushButton_581.clicked.connect(lambda: add_new_node_to_triplet(77))
window.pushButton_582.clicked.connect(lambda: add_new_node_to_triplet(78))
window.pushButton_583.clicked.connect(lambda: add_new_node_to_triplet(79))
window.pushButton_584.clicked.connect(lambda: add_new_node_to_triplet(80))
window.pushButton_585.clicked.connect(lambda: add_new_node_to_triplet(81))
window.pushButton_586.clicked.connect(lambda: add_new_node_to_triplet(82))
window.pushButton_587.clicked.connect(lambda: add_new_node_to_triplet(83))
window.pushButton_588.clicked.connect(lambda: add_new_node_to_triplet(84))

# Complete tree N=121
window.pushButton_589.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_590.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_591.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_592.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_593.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_594.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_595.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_596.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_597.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_598.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_599.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_600.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_601.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_602.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_603.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_604.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_605.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_606.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_607.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_608.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_609.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_610.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_611.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_612.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_613.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_614.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_615.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_616.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_617.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_618.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_619.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_620.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_621.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_622.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_623.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_624.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_625.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_626.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_627.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_628.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_629.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_630.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_631.clicked.connect(lambda: add_new_node_to_triplet(42))
window.pushButton_632.clicked.connect(lambda: add_new_node_to_triplet(43))
window.pushButton_633.clicked.connect(lambda: add_new_node_to_triplet(44))
window.pushButton_634.clicked.connect(lambda: add_new_node_to_triplet(45))
window.pushButton_635.clicked.connect(lambda: add_new_node_to_triplet(46))
window.pushButton_636.clicked.connect(lambda: add_new_node_to_triplet(47))
window.pushButton_637.clicked.connect(lambda: add_new_node_to_triplet(48))
window.pushButton_638.clicked.connect(lambda: add_new_node_to_triplet(49))
window.pushButton_639.clicked.connect(lambda: add_new_node_to_triplet(50))
window.pushButton_640.clicked.connect(lambda: add_new_node_to_triplet(51))
window.pushButton_641.clicked.connect(lambda: add_new_node_to_triplet(52))
window.pushButton_642.clicked.connect(lambda: add_new_node_to_triplet(53))
window.pushButton_643.clicked.connect(lambda: add_new_node_to_triplet(54))
window.pushButton_644.clicked.connect(lambda: add_new_node_to_triplet(55))
window.pushButton_645.clicked.connect(lambda: add_new_node_to_triplet(56))
window.pushButton_646.clicked.connect(lambda: add_new_node_to_triplet(57))
window.pushButton_647.clicked.connect(lambda: add_new_node_to_triplet(58))
window.pushButton_648.clicked.connect(lambda: add_new_node_to_triplet(59))
window.pushButton_649.clicked.connect(lambda: add_new_node_to_triplet(60))
window.pushButton_650.clicked.connect(lambda: add_new_node_to_triplet(61))
window.pushButton_651.clicked.connect(lambda: add_new_node_to_triplet(62))
window.pushButton_652.clicked.connect(lambda: add_new_node_to_triplet(63))
window.pushButton_653.clicked.connect(lambda: add_new_node_to_triplet(64))
window.pushButton_654.clicked.connect(lambda: add_new_node_to_triplet(65))
window.pushButton_655.clicked.connect(lambda: add_new_node_to_triplet(66))
window.pushButton_656.clicked.connect(lambda: add_new_node_to_triplet(67))
window.pushButton_657.clicked.connect(lambda: add_new_node_to_triplet(68))
window.pushButton_658.clicked.connect(lambda: add_new_node_to_triplet(69))
window.pushButton_659.clicked.connect(lambda: add_new_node_to_triplet(70))
window.pushButton_660.clicked.connect(lambda: add_new_node_to_triplet(71))
window.pushButton_661.clicked.connect(lambda: add_new_node_to_triplet(72))
window.pushButton_662.clicked.connect(lambda: add_new_node_to_triplet(73))
window.pushButton_663.clicked.connect(lambda: add_new_node_to_triplet(74))
window.pushButton_664.clicked.connect(lambda: add_new_node_to_triplet(75))
window.pushButton_665.clicked.connect(lambda: add_new_node_to_triplet(76))
window.pushButton_666.clicked.connect(lambda: add_new_node_to_triplet(77))
window.pushButton_667.clicked.connect(lambda: add_new_node_to_triplet(78))
window.pushButton_668.clicked.connect(lambda: add_new_node_to_triplet(79))
window.pushButton_669.clicked.connect(lambda: add_new_node_to_triplet(80))
window.pushButton_670.clicked.connect(lambda: add_new_node_to_triplet(81))
window.pushButton_671.clicked.connect(lambda: add_new_node_to_triplet(82))
window.pushButton_672.clicked.connect(lambda: add_new_node_to_triplet(83))
window.pushButton_673.clicked.connect(lambda: add_new_node_to_triplet(84))
window.pushButton_674.clicked.connect(lambda: add_new_node_to_triplet(85))
window.pushButton_675.clicked.connect(lambda: add_new_node_to_triplet(86))
window.pushButton_676.clicked.connect(lambda: add_new_node_to_triplet(87))
window.pushButton_677.clicked.connect(lambda: add_new_node_to_triplet(88))
window.pushButton_678.clicked.connect(lambda: add_new_node_to_triplet(89))
window.pushButton_679.clicked.connect(lambda: add_new_node_to_triplet(90))
window.pushButton_680.clicked.connect(lambda: add_new_node_to_triplet(91))
window.pushButton_681.clicked.connect(lambda: add_new_node_to_triplet(92))
window.pushButton_682.clicked.connect(lambda: add_new_node_to_triplet(93))
window.pushButton_683.clicked.connect(lambda: add_new_node_to_triplet(94))
window.pushButton_684.clicked.connect(lambda: add_new_node_to_triplet(95))
window.pushButton_685.clicked.connect(lambda: add_new_node_to_triplet(96))
window.pushButton_686.clicked.connect(lambda: add_new_node_to_triplet(97))
window.pushButton_687.clicked.connect(lambda: add_new_node_to_triplet(98))
window.pushButton_688.clicked.connect(lambda: add_new_node_to_triplet(99))
window.pushButton_689.clicked.connect(lambda: add_new_node_to_triplet(100))
window.pushButton_690.clicked.connect(lambda: add_new_node_to_triplet(101))
window.pushButton_691.clicked.connect(lambda: add_new_node_to_triplet(102))
window.pushButton_692.clicked.connect(lambda: add_new_node_to_triplet(103))
window.pushButton_693.clicked.connect(lambda: add_new_node_to_triplet(104))
window.pushButton_694.clicked.connect(lambda: add_new_node_to_triplet(105))
window.pushButton_695.clicked.connect(lambda: add_new_node_to_triplet(106))
window.pushButton_696.clicked.connect(lambda: add_new_node_to_triplet(107))
window.pushButton_697.clicked.connect(lambda: add_new_node_to_triplet(108))
window.pushButton_698.clicked.connect(lambda: add_new_node_to_triplet(109))
window.pushButton_699.clicked.connect(lambda: add_new_node_to_triplet(110))
window.pushButton_700.clicked.connect(lambda: add_new_node_to_triplet(111))
window.pushButton_701.clicked.connect(lambda: add_new_node_to_triplet(112))
window.pushButton_702.clicked.connect(lambda: add_new_node_to_triplet(113))
window.pushButton_703.clicked.connect(lambda: add_new_node_to_triplet(114))
window.pushButton_704.clicked.connect(lambda: add_new_node_to_triplet(115))
window.pushButton_705.clicked.connect(lambda: add_new_node_to_triplet(116))
window.pushButton_706.clicked.connect(lambda: add_new_node_to_triplet(117))
window.pushButton_707.clicked.connect(lambda: add_new_node_to_triplet(118))
window.pushButton_708.clicked.connect(lambda: add_new_node_to_triplet(119))
window.pushButton_709.clicked.connect(lambda: add_new_node_to_triplet(120))

# Complete tree N=127
window.pushButton_719.clicked.connect(lambda: add_new_node_to_triplet(0))
window.pushButton_720.clicked.connect(lambda: add_new_node_to_triplet(1))
window.pushButton_721.clicked.connect(lambda: add_new_node_to_triplet(2))
window.pushButton_722.clicked.connect(lambda: add_new_node_to_triplet(3))
window.pushButton_723.clicked.connect(lambda: add_new_node_to_triplet(4))
window.pushButton_724.clicked.connect(lambda: add_new_node_to_triplet(5))
window.pushButton_725.clicked.connect(lambda: add_new_node_to_triplet(6))
window.pushButton_726.clicked.connect(lambda: add_new_node_to_triplet(7))
window.pushButton_727.clicked.connect(lambda: add_new_node_to_triplet(8))
window.pushButton_728.clicked.connect(lambda: add_new_node_to_triplet(9))
window.pushButton_729.clicked.connect(lambda: add_new_node_to_triplet(10))
window.pushButton_730.clicked.connect(lambda: add_new_node_to_triplet(11))
window.pushButton_731.clicked.connect(lambda: add_new_node_to_triplet(12))
window.pushButton_732.clicked.connect(lambda: add_new_node_to_triplet(13))
window.pushButton_733.clicked.connect(lambda: add_new_node_to_triplet(14))
window.pushButton_734.clicked.connect(lambda: add_new_node_to_triplet(15))
window.pushButton_735.clicked.connect(lambda: add_new_node_to_triplet(16))
window.pushButton_736.clicked.connect(lambda: add_new_node_to_triplet(17))
window.pushButton_737.clicked.connect(lambda: add_new_node_to_triplet(18))
window.pushButton_738.clicked.connect(lambda: add_new_node_to_triplet(19))
window.pushButton_739.clicked.connect(lambda: add_new_node_to_triplet(20))
window.pushButton_740.clicked.connect(lambda: add_new_node_to_triplet(21))
window.pushButton_741.clicked.connect(lambda: add_new_node_to_triplet(22))
window.pushButton_742.clicked.connect(lambda: add_new_node_to_triplet(23))
window.pushButton_743.clicked.connect(lambda: add_new_node_to_triplet(24))
window.pushButton_744.clicked.connect(lambda: add_new_node_to_triplet(25))
window.pushButton_745.clicked.connect(lambda: add_new_node_to_triplet(26))
window.pushButton_746.clicked.connect(lambda: add_new_node_to_triplet(27))
window.pushButton_747.clicked.connect(lambda: add_new_node_to_triplet(28))
window.pushButton_748.clicked.connect(lambda: add_new_node_to_triplet(29))
window.pushButton_749.clicked.connect(lambda: add_new_node_to_triplet(30))
window.pushButton_750.clicked.connect(lambda: add_new_node_to_triplet(31))
window.pushButton_751.clicked.connect(lambda: add_new_node_to_triplet(32))
window.pushButton_752.clicked.connect(lambda: add_new_node_to_triplet(33))
window.pushButton_753.clicked.connect(lambda: add_new_node_to_triplet(34))
window.pushButton_754.clicked.connect(lambda: add_new_node_to_triplet(35))
window.pushButton_755.clicked.connect(lambda: add_new_node_to_triplet(36))
window.pushButton_756.clicked.connect(lambda: add_new_node_to_triplet(37))
window.pushButton_757.clicked.connect(lambda: add_new_node_to_triplet(38))
window.pushButton_758.clicked.connect(lambda: add_new_node_to_triplet(39))
window.pushButton_759.clicked.connect(lambda: add_new_node_to_triplet(40))
window.pushButton_760.clicked.connect(lambda: add_new_node_to_triplet(41))
window.pushButton_761.clicked.connect(lambda: add_new_node_to_triplet(42))
window.pushButton_762.clicked.connect(lambda: add_new_node_to_triplet(43))
window.pushButton_763.clicked.connect(lambda: add_new_node_to_triplet(44))
window.pushButton_764.clicked.connect(lambda: add_new_node_to_triplet(45))
window.pushButton_765.clicked.connect(lambda: add_new_node_to_triplet(46))
window.pushButton_766.clicked.connect(lambda: add_new_node_to_triplet(47))
window.pushButton_767.clicked.connect(lambda: add_new_node_to_triplet(48))
window.pushButton_768.clicked.connect(lambda: add_new_node_to_triplet(49))
window.pushButton_769.clicked.connect(lambda: add_new_node_to_triplet(50))
window.pushButton_770.clicked.connect(lambda: add_new_node_to_triplet(51))
window.pushButton_771.clicked.connect(lambda: add_new_node_to_triplet(52))
window.pushButton_772.clicked.connect(lambda: add_new_node_to_triplet(53))
window.pushButton_773.clicked.connect(lambda: add_new_node_to_triplet(54))
window.pushButton_774.clicked.connect(lambda: add_new_node_to_triplet(55))
window.pushButton_775.clicked.connect(lambda: add_new_node_to_triplet(56))
window.pushButton_776.clicked.connect(lambda: add_new_node_to_triplet(57))
window.pushButton_777.clicked.connect(lambda: add_new_node_to_triplet(58))
window.pushButton_778.clicked.connect(lambda: add_new_node_to_triplet(59))
window.pushButton_779.clicked.connect(lambda: add_new_node_to_triplet(60))
window.pushButton_780.clicked.connect(lambda: add_new_node_to_triplet(61))
window.pushButton_781.clicked.connect(lambda: add_new_node_to_triplet(62))
window.pushButton_782.clicked.connect(lambda: add_new_node_to_triplet(63))
window.pushButton_783.clicked.connect(lambda: add_new_node_to_triplet(64))
window.pushButton_784.clicked.connect(lambda: add_new_node_to_triplet(65))
window.pushButton_785.clicked.connect(lambda: add_new_node_to_triplet(66))
window.pushButton_786.clicked.connect(lambda: add_new_node_to_triplet(67))
window.pushButton_787.clicked.connect(lambda: add_new_node_to_triplet(68))
window.pushButton_788.clicked.connect(lambda: add_new_node_to_triplet(69))
window.pushButton_789.clicked.connect(lambda: add_new_node_to_triplet(70))
window.pushButton_790.clicked.connect(lambda: add_new_node_to_triplet(71))
window.pushButton_791.clicked.connect(lambda: add_new_node_to_triplet(72))
window.pushButton_792.clicked.connect(lambda: add_new_node_to_triplet(73))
window.pushButton_793.clicked.connect(lambda: add_new_node_to_triplet(74))
window.pushButton_794.clicked.connect(lambda: add_new_node_to_triplet(75))
window.pushButton_795.clicked.connect(lambda: add_new_node_to_triplet(76))
window.pushButton_796.clicked.connect(lambda: add_new_node_to_triplet(77))
window.pushButton_797.clicked.connect(lambda: add_new_node_to_triplet(78))
window.pushButton_798.clicked.connect(lambda: add_new_node_to_triplet(79))
window.pushButton_799.clicked.connect(lambda: add_new_node_to_triplet(80))
window.pushButton_800.clicked.connect(lambda: add_new_node_to_triplet(81))
window.pushButton_801.clicked.connect(lambda: add_new_node_to_triplet(82))
window.pushButton_802.clicked.connect(lambda: add_new_node_to_triplet(83))
window.pushButton_803.clicked.connect(lambda: add_new_node_to_triplet(84))
window.pushButton_804.clicked.connect(lambda: add_new_node_to_triplet(85))
window.pushButton_805.clicked.connect(lambda: add_new_node_to_triplet(86))
window.pushButton_806.clicked.connect(lambda: add_new_node_to_triplet(87))
window.pushButton_807.clicked.connect(lambda: add_new_node_to_triplet(88))
window.pushButton_808.clicked.connect(lambda: add_new_node_to_triplet(89))
window.pushButton_809.clicked.connect(lambda: add_new_node_to_triplet(90))
window.pushButton_810.clicked.connect(lambda: add_new_node_to_triplet(91))
window.pushButton_811.clicked.connect(lambda: add_new_node_to_triplet(92))
window.pushButton_812.clicked.connect(lambda: add_new_node_to_triplet(93))
window.pushButton_813.clicked.connect(lambda: add_new_node_to_triplet(94))
window.pushButton_814.clicked.connect(lambda: add_new_node_to_triplet(95))
window.pushButton_815.clicked.connect(lambda: add_new_node_to_triplet(96))
window.pushButton_816.clicked.connect(lambda: add_new_node_to_triplet(97))
window.pushButton_817.clicked.connect(lambda: add_new_node_to_triplet(98))
window.pushButton_818.clicked.connect(lambda: add_new_node_to_triplet(99))
window.pushButton_819.clicked.connect(lambda: add_new_node_to_triplet(100))
window.pushButton_820.clicked.connect(lambda: add_new_node_to_triplet(101))
window.pushButton_821.clicked.connect(lambda: add_new_node_to_triplet(102))
window.pushButton_822.clicked.connect(lambda: add_new_node_to_triplet(103))
window.pushButton_823.clicked.connect(lambda: add_new_node_to_triplet(104))
window.pushButton_824.clicked.connect(lambda: add_new_node_to_triplet(105))
window.pushButton_825.clicked.connect(lambda: add_new_node_to_triplet(106))
window.pushButton_826.clicked.connect(lambda: add_new_node_to_triplet(107))
window.pushButton_827.clicked.connect(lambda: add_new_node_to_triplet(108))
window.pushButton_828.clicked.connect(lambda: add_new_node_to_triplet(109))
window.pushButton_829.clicked.connect(lambda: add_new_node_to_triplet(110))
window.pushButton_830.clicked.connect(lambda: add_new_node_to_triplet(111))
window.pushButton_831.clicked.connect(lambda: add_new_node_to_triplet(112))
window.pushButton_832.clicked.connect(lambda: add_new_node_to_triplet(113))
window.pushButton_833.clicked.connect(lambda: add_new_node_to_triplet(114))
window.pushButton_834.clicked.connect(lambda: add_new_node_to_triplet(115))
window.pushButton_835.clicked.connect(lambda: add_new_node_to_triplet(116))
window.pushButton_836.clicked.connect(lambda: add_new_node_to_triplet(117))
window.pushButton_837.clicked.connect(lambda: add_new_node_to_triplet(118))
window.pushButton_838.clicked.connect(lambda: add_new_node_to_triplet(119))
window.pushButton_839.clicked.connect(lambda: add_new_node_to_triplet(120))
window.pushButton_840.clicked.connect(lambda: add_new_node_to_triplet(121))
window.pushButton_841.clicked.connect(lambda: add_new_node_to_triplet(122))
window.pushButton_842.clicked.connect(lambda: add_new_node_to_triplet(123))
window.pushButton_843.clicked.connect(lambda: add_new_node_to_triplet(124))
window.pushButton_844.clicked.connect(lambda: add_new_node_to_triplet(125))
window.pushButton_845.clicked.connect(lambda: add_new_node_to_triplet(126))










window.show()
window2.show()
app.exec()