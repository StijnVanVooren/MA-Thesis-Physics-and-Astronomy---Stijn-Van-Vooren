import sys
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtGui

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

ind30 = [16,3,14,19,2,8,5,9,30,7,17,20,10,18,6,29,26,21,4,28,1,25,15,24,23,22,12,27,11,13]
print(ind30[5])

plt.close('all')


loader = QUiLoader();

app = QtWidgets.QApplication(sys.argv)
window = loader.load("form.ui", None) # Loads UI, but at RUNTIME!!

d1=15
d2=15
t=15
data_path =dirname(abspath(__file__))
print(data_path)
save_path = ""

def load(filename,N):
    global N_trials,T_relax,T_evolve,J_min,J_max,J_res,runtime
    # Initialize parameters:    
    # load file INFO    
    filename = data_path+"/Data/"+filename
    currentLine = 1
    with open(filename + "_INFO.txt") as textFile:
        for line in textFile:
                if(currentLine == 2):
                    N_trials = int(line)
                elif(currentLine == 5):
                    T_relax = int(line)
                elif(currentLine == 8):
                    T_evolve = int(line)
                elif(currentLine == 11):
                    J_min = float(line)
                elif(currentLine == 14):
                    J_max = float(line)
                elif(currentLine == 17):
                    J_res = int(line)
                elif(currentLine == 20):
                    runtime = float(line)

                currentLine+=1
    textFile.close()

    # compute J values that were simulated out of J_min, J_max and J_res:
    global dJ
    dJ = (J_max-J_min)/(J_res-1.0);
    global J
    J = np.empty(shape=J_res, dtype=float);
    for i in range(J_res):
        J[i] = J_min + i*dJ;
        
    # make runtime string for title in plot (convert to either hours, minutes, second or microseconds):
    global runtime_string
    if(runtime < 0.001):
        runtime_string = str(round(runtime*1000,2))+" ms"
    elif(runtime < 60):
        runtime_string = str(round(runtime,2))+" s"
    elif(runtime < 3600):
        runtime_string = str(round(runtime/60,2))+ " m"
    elif(runtime >=3600):
        runtime_string = str(round(runtime/3600,2)) + " h"    
    

    #load Chi matrix, a comma-delimited text file into an np matrix

    resultList = []
    with open(filename + "_Chi.txt") as textFile:
        for line in textFile:
            line = line.rstrip('\n')  # "3.0,4.0,5.0"
            stringVals = line.split(' ')   # ["3.0", "4.0, "5.0"]
            del stringVals[-1]; #to get rid of last element ""
            floatVals = list(map(np.float32, stringVals))  # [3.0, 4.0, 5.0]
            resultList.append(floatVals)  # [[3.0, 4.0, 5.0] , [6.0, 7.0, 8.0]]
    textFile.close()
    global Chi
    Chi = np.asarray(resultList, dtype=np.float32)

    #load S_T matrix

    # Create an empty 4-dimensional matrix S_T
    global S_T
    S_T = np.zeros((N, N, N, J_res))

    # Loop over the text files
    for t in range(N):        
        with open(filename+"_ST_"+str(t)+".txt", "r") as f:
            # Loop over the lines in the file
            for i, line in enumerate(f):
                if i % 2 == 0:
                    # Parse the sources from the line
                    d1, d2 = map(int, line.strip().split())
                else:
                    # Parse the float values and store them in S_T                    
                    S_T[d1, d2, t, :] = np.array(line.strip().split(), dtype=float)
                    S_T[d2, d1, t, :] = np.array(line.strip().split(), dtype=float)

N = 30

def loadRandomTreeN30():
    global N
    N = 30
    load("Tree30_fullspec_VSC_metropolis",30)

    min = 1
    window.spinBox.setMinimum(min)
    window.spinBox_2.setMinimum(min)
    window.spinBox_3.setMinimum(min)

    max = 30
    window.spinBox.setMaximum(max)
    window.spinBox_2.setMaximum(max)
    window.spinBox_3.setMaximum(max)

loadRandomTreeN30()
meanChi = np.mean(Chi, axis=1);

def setd1(new_value):
    global d1
    global ind30
    d1 = ind30[new_value-1]-1    

window.spinBox.valueChanged.connect(setd1)

def setd2(new_value):
    global d2
    global ind30
    d2 = ind30[new_value-1]-1    

window.spinBox_2.valueChanged.connect(setd2)

def sett(new_value):
    global t
    global ind30
    t = ind30[new_value-1]-1    
window.spinBox_3.valueChanged.connect(sett)

setting_value =  1
def setNextValue(new_value):
    global setting_value
    if(setting_value == 1):
        window.spinBox.setValue(new_value)
        #d1 = new_value
    elif(setting_value == 2):
        window.spinBox_2.setValue(new_value)
        #d2 = new_value
    else:
        window.spinBox_3.setValue(new_value)
        #t = new_value
        setting_value = 0
    setting_value += 1

def d1_button_clicked():
    global setting_value
    setting_value = 1

window.pushButton_31.clicked.connect(d1_button_clicked)

def d2_button_clicked():
    global setting_value
    setting_value = 2

window.pushButton_34.clicked.connect(d2_button_clicked)

def t_button_clicked():
    global setting_value
    setting_value = 3

window.pushButton_32.clicked.connect(t_button_clicked)

def setNextValue1():
    setNextValue(1)
window.pushButton.clicked.connect(setNextValue1)

def setNextValue2():
    setNextValue(2)
window.pushButton_2.clicked.connect(setNextValue2)

def setNextValue3():
    setNextValue(3)
window.pushButton_3.clicked.connect(setNextValue3)

def setNextValue4():
    setNextValue(4)
window.pushButton_4.clicked.connect(setNextValue4)

def setNextValue5():
    setNextValue(5)
window.pushButton_5.clicked.connect(setNextValue5)

def setNextValue6():
    setNextValue(6)
window.pushButton_6.clicked.connect(setNextValue6)

def setNextValue7():
    setNextValue(7)
window.pushButton_7.clicked.connect(setNextValue7)

def setNextValue8():
    setNextValue(8)
window.pushButton_8.clicked.connect(setNextValue8)

def setNextValue9():
    setNextValue(9)
window.pushButton_9.clicked.connect(setNextValue9)

def setNextValue10():
    setNextValue(10)
window.pushButton_10.clicked.connect(setNextValue10)

def setNextValue11():
    setNextValue(11)
window.pushButton_11.clicked.connect(setNextValue11)

def setNextValue12():
    setNextValue(12)
window.pushButton_12.clicked.connect(setNextValue12)

def setNextValue13():
    setNextValue(13)
window.pushButton_13.clicked.connect(setNextValue13)

def setNextValue14():
    setNextValue(14)
window.pushButton_14.clicked.connect(setNextValue14)

def setNextValue15():
    setNextValue(15)
window.pushButton_15.clicked.connect(setNextValue15)

def setNextValue16():
    setNextValue(16)
window.pushButton_16.clicked.connect(setNextValue16)

def setNextValue17():
    setNextValue(17)
window.pushButton_17.clicked.connect(setNextValue17)

def setNextValue18():
    setNextValue(18)
window.pushButton_18.clicked.connect(setNextValue18)

def setNextValue19():
    setNextValue(19)
window.pushButton_19.clicked.connect(setNextValue19)

def setNextValue20():
    setNextValue(20)
window.pushButton_20.clicked.connect(setNextValue20)

def setNextValue21():
    setNextValue(21)
window.pushButton_21.clicked.connect(setNextValue21)

def setNextValue22():
    setNextValue(22)
window.pushButton_22.clicked.connect(setNextValue22)

def setNextValue23():
    setNextValue(23)
window.pushButton_23.clicked.connect(setNextValue23)

def setNextValue24():
    setNextValue(24)
window.pushButton_24.clicked.connect(setNextValue24)

def setNextValue25():
    setNextValue(25)
window.pushButton_25.clicked.connect(setNextValue25)

def setNextValue26():
    setNextValue(26)
window.pushButton_26.clicked.connect(setNextValue26)

def setNextValue27():
    setNextValue(27)
window.pushButton_27.clicked.connect(setNextValue27)

def setNextValue28():
    setNextValue(28)
window.pushButton_28.clicked.connect(setNextValue28)

def setNextValue29():
    setNextValue(29)
window.pushButton_29.clicked.connect(setNextValue29)

def setNextValue30():
    setNextValue(30)
window.pushButton_30.clicked.connect(setNextValue30)

window.checkBox_4.setChecked(True)
window.checkBox_5.setChecked(True)
window.checkBox_6.setChecked(True)
window.radioButton.toggle()

plot_type = 1
average_over_d1 = False
average_over_d2 = False
average_over_t = False
fix_d1 = True
fix_d2 = True
fix_t = True

color_cycle = plt.cm.Set1(np.linspace(0, 1, 9))

def set_plot_type_1(boool):
    global plot_type
    if(boool):
        plot_type = 1  
        # reset the color cycle to default
        plt.gca().set_prop_cycle(None)  
window.radioButton.toggled.connect(set_plot_type_1)

def set_plot_type_2(boool):
    global plot_type
    if(boool):
        plot_type = 2 
        plt.gca().set_prop_cycle(plt.cycler('color', color_cycle))  
window.radioButton_2.toggled.connect(set_plot_type_2)

def set_plot_type_3(boool):
    global plot_type
    if(boool):
        plot_type = 3  
        # reset the color cycle to default
        plt.gca().set_prop_cycle(None) 
window.radioButton_3.toggled.connect(set_plot_type_3)

def set_plot_type_4(boool):
    global plot_type
    if(boool):
        plot_type = 4   
        # reset the color cycle to default
        plt.gca().set_prop_cycle(None) 
window.radioButton_4.toggled.connect(set_plot_type_4)


def print_bool_state():
    print("Average over d1 = ", average_over_d1)    
    print("Average over d2 = ", average_over_d2)    
    print("Average over t = ", average_over_t)    
    print("Fix d1 = ", fix_d1)    
    print("Fix d2 = ", fix_d2)    
    print("Fix t = ", fix_t)
    

def flip_average_over_d1(new_bool):
    global fix_d1
    global average_over_d1
    average_over_d1 = new_bool  
    window.checkBox_4.blockSignals(True)
    window.checkBox_4.setChecked(False)
    window.checkBox_4.blockSignals(False)  
    fix_d1 = False
    #print_bool_state()
window.checkBox.toggled.connect(flip_average_over_d1)

def flip_average_over_d2(new_bool):
    global average_over_d2
    global fix_d2
    average_over_d2 = new_bool  
    window.checkBox_5.blockSignals(True)
    window.checkBox_5.setChecked(False)
    window.checkBox_5.blockSignals(False)  
    fix_d2 = False 
    #print_bool_state()
window.checkBox_2.toggled.connect(flip_average_over_d2)

def flip_average_over_t(new_bool):
    global average_over_t
    global fix_t
    average_over_t = new_bool   
    window.checkBox_6.blockSignals(True)
    window.checkBox_6.setChecked(False)
    window.checkBox_6.blockSignals(False)  
    fix_t = False
    #print_bool_state()
window.checkBox_3.toggled.connect(flip_average_over_t)

def flip_fix_d1(new_bool):
    global fix_d1
    global average_over_d1
    fix_d1 = new_bool    
    window.checkBox.blockSignals(True)
    window.checkBox.setChecked(False)
    window.checkBox.blockSignals(False)  
    average_over_d1 = False
    #print_bool_state()
window.checkBox_4.toggled.connect(flip_fix_d1)

def flip_fix_d2(new_bool):
    global fix_d2
    global average_over_d2
    fix_d2 = new_bool 
    window.checkBox_2.blockSignals(True)
    window.checkBox_2.setChecked(False)
    window.checkBox_2.blockSignals(False)  
    average_over_d2 = False 
    #print_bool_state()
window.checkBox_5.toggled.connect(flip_fix_d2)

def flip_fix_t(new_bool):
    global fix_t
    global average_over_t
    fix_t = new_bool  
    window.checkBox_3.blockSignals(True)
    window.checkBox_3.setChecked(False)
    window.checkBox_3.blockSignals(False)  
    average_over_t = False
    #print_bool_state()
window.checkBox_6.toggled.connect(flip_fix_t)

global fig
global ax 
fig, ax = plt.subplots()
plt.ion()  # Turn on interactive mode

def plot_type_1(Y,scaling):
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$S_T,\chi$")
    plt.title(r"$S_T$"+" and "+r"$\chi$"+" VS "+r"$\beta$") 
    #ax.set_aspect('equal')     
    plt.plot(J,meanChi*scaling)#,label = r"$\chi$")
    plt.scatter(J,Y,label = r"$S_T($"+str(window.spinBox.value())+","+str(window.spinBox_2.value())+r"$\rightarrow$"+str(window.spinBox_3.value())+")") 

def plot_type_2(Y,scaling):
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$S_T$"+" fit"+r"$,\chi$")
    plt.title(r"$S_T$"+" and "+r"$\chi$"+" VS "+r"$\beta$") 
    #Fit ST:
    # Create polynomial features of degree 20
    poly = PolynomialFeatures(degree=30)
    X_poly = poly.fit_transform(J.reshape(-1, 1))
    # Fit the polynomial regression model
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, Y)
    # Predict the response for a sequence of X values
    X_seq = np.linspace(J_min, J_max, num=400).reshape(-1, 1)
    X_seq_poly = poly.fit_transform(X_seq)
    Y_seq = poly_reg.predict(X_seq_poly)
    #Plot the original data and the predicted values    
    plt.plot(X_seq, Y_seq,label = r"$S_T($"+str(window.spinBox.value())+","+str(window.spinBox_2.value())+r"$\rightarrow$"+str(window.spinBox_3.value())+")")
    plt.plot(J,meanChi*scaling)#,label = r"$\chi$") 



def plot_type_4(Y,scaling):
    plt.xlabel(r"$\beta$")
    plt.ylabel("max "+r"$S_T,\chi$")
    plt.title(r"$S_T$"+" and "+r"$\chi$"+" VS "+r"$\beta$") 
    # create polynomial features up to degree 20
    poly = PolynomialFeatures(degree=30)

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

    # plot the data, the fitted curve, and the maximum point
    #plt.plot(J, Y, 'ro')
    #plt.plot(X_grid, Y_pred, 'b-')
    plt.plot(*max_point,'x' ,label = "max "+r"$S_T($"+str(window.spinBox.value())+","+str(window.spinBox_2.value())+r"$\rightarrow$"+str(window.spinBox_3.value())+")")
    plt.plot(J,meanChi*scaling)#,label = r"$\chi$") 

def plot_any(Y,scaling):
    if(plot_type == 1):  
        plot_type_1(Y,scaling)     
    if(plot_type == 2):
        plot_type_2(Y,scaling)     
    if(plot_type == 3):
        plot_type_3(Y,scaling)
    if(plot_type == 4):
        plot_type_4(Y,scaling)


def plot():
    global fig
    global ax
    global d1
    global d2
    global t
    global meanChi
    global J
    global N
    
    
    #Reopen figure if it is closed
    if(plt.get_fignums()):
        plt.show(block=False)
        # Draw horizontal line
        plt.axhline(0)

    #Go over all different possible combinations of averging, fixing and plot types:
    if fix_d1 and fix_d2 and fix_t:
        Y = S_T[d1][d2][t]        
    elif fix_d1 and fix_d2:
        Y = S_T[d1, d2, ...]  
        if(average_over_t):
            Y = np.mean(Y,axis = 0)             
    elif fix_d1 and fix_t:
        Y = S_T[d1, :, t, :]  
        if(average_over_d2):
            Y = np.mean(Y,axis = 0)        
    elif fix_d2 and fix_t:
        Y = S_T[:, d2, t, :]
        if(average_over_d1):
            Y = np.mean(Y,axis = 0)  
    elif fix_d1:
        Y = S_T[d1, ...]
        if(average_over_d2 and average_over_t):
            Y = np.mean(Y,axis = (0,1)) 
        elif(average_over_d2):
            Y = np.mean(Y,axis = 0) 
        elif(average_over_t):
            Y = np.mean(Y,axis = 1)
    elif fix_d2:
        Y = S_T[:, d2, ...]
        if(average_over_d1 and average_over_t):
            Y = np.mean(Y,axis = (0,1)) 
        elif(average_over_d1):
            Y = np.mean(Y,axis = 0) 
        elif(average_over_t):
            Y = np.mean(Y,axis = 1)
    elif fix_t:
        Y = S_T[:, :, t, :]
        if(average_over_d1 and average_over_d2):
            Y = np.mean(Y,axis = (0,1)) 
        elif(average_over_d1):
            Y = np.mean(Y,axis = 0) 
        elif(average_over_d2):
            Y = np.mean(Y,axis = 1)
    else:
        axes_to_avg = []
        if average_over_d1 and not fix_d1:
            axes_to_avg.append(0)
        if average_over_d2 and not fix_d2:
            axes_to_avg.append(1)
        if average_over_t and not fix_t:
            axes_to_avg.append(2)
        Y = np.mean(S_T, axis=tuple(axes_to_avg))
             
    #Perform the actual plot:    
    scaling = np.max(Y)/np.max(meanChi)
    if Y.ndim == 1:
        plot_any(Y, scaling)

    elif Y.ndim == 2:
        for ind1 in range(N):
            plot_any(Y[ind1,:],scaling)

    elif Y.ndim == 3:
        if(fix_t):
            for ind1 in range(N):
                for ind2 in range(ind1,N):
                    plot_any(Y[ind1,ind2,:],scaling)
        else: 
            for ind1 in range(N):
                for ind2 in range(N):
                    if ind1 != ind2:
                        plot_any(Y[ind1,ind2,:],scaling)
    
    elif Y.ndim == 4:
        for ind1 in range(N):
            for ind2 in range(ind1,N):
                for ind3 in range(N):
                    plot_any(Y[ind1,ind2,ind3,:],scaling)

        
  

    plt.legend()
    fig.canvas.draw()

window.pushButton_33.clicked.connect(plot)

def clear_plot():    
    global fig
    global ax
    fig.clear()
    ax.clear()
    # Draw horizontal line
    plt.axhline(0)
    fig.canvas.draw()
window.pushButton_35.clicked.connect(clear_plot)


window.show()
app.exec()
