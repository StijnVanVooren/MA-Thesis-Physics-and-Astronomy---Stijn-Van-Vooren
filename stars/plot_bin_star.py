from os.path import dirname, abspath
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import r2_score

from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def fit_polynomial_and_find_max_x(x_data, y_data, degree):
    # Generate polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x_data.reshape(-1, 1))

    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(x_poly, y_data)

    # Generate a finer x range for more accurate interpolation
    x_range = np.linspace(min(x_data), max(x_data), 10000)
    x_range_poly = polynomial_features.transform(x_range.reshape(-1, 1))

    # Evaluate the polynomial fit over the x range
    y_fit = model.predict(x_range_poly)

    # Find the index of the maximum value in the fit
    max_index = np.argmax(y_fit)

    # Get the x value corresponding to the maximum value
    max_x = x_range[max_index]

    # Calculate the R-squared value using the validation dataset
    x_validation_poly = polynomial_features.transform(x_validation.reshape(-1, 1))
    y_validation_pred = model.predict(x_validation_poly)
    r_squared = r2_score(y_validation, y_validation_pred)

    # Plot the original data and the fit
    #plt.scatter(x_data, y_data, label='Original Data')
    #plt.plot(x_range, y_fit, color='red', label='Polynomial Fit')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.legend()
    #plt.title('Polynomial Fit')
    #plt.show()

    return max_x


data_path = dirname(abspath(__file__))

def plot_bin_star(driver,target,dJ):
    extension = str(driver)+"_"+str(target)
    filename = data_path+"\Data\BinaryTrees\BinaryTree_"
    filenameChi = filename+"Chi_"+extension
    filenameTE1 = filename+"TE1_"+extension
    filenameTE2 = filename+"TE2_"+extension

    Chi = np.loadtxt(filenameChi)
    TE1 = np.loadtxt(filenameTE1)
    TE2 = np.loadtxt(filenameTE2)

    J = np.linspace(dJ, dJ * TE1.size, TE1.size)

    # Plotting Chi, TE1, and TE2 vs J on the same plot
    #plt.plot(J, Chi, label='Chi')
    plt.scatter(J, TE1, label='TE1')
    #plt.plot(J, TE2, label='TE2')


def plot_bin_star_normalised(driver,target,dJ):
    extension = str(driver)+"_"+str(target)
    filename = data_path+"\Data\BinaryTrees\BinaryTree_"
    filenameChi = filename+"Chi_"+extension
    filenameTE1 = filename+"TE1_"+extension
    filenameTE2 = filename+"TE2_"+extension

    Chi = np.loadtxt(filenameChi)
    TE1 = np.loadtxt(filenameTE1)
    TE2 = np.loadtxt(filenameTE2)

    J = np.linspace(dJ, dJ * TE1.size, TE1.size)

    max_Chi = np.max(Chi)
    max_TE1 = np.max(TE1)
    max_TE2 = np.max(TE2)
    Chi = Chi/max_Chi
    TE1 = TE1/max_TE1
    TE2 = TE2/max_TE2

def TE1_normalised(driver,target,dJ):
    extension = str(driver)+"_"+str(target)
    filename = data_path+"\Data\BinaryTrees\BinaryTree_"

    filenameTE1 = filename+"TE1_"+extension

    TE1 = np.loadtxt(filenameTE1)

    J = np.linspace(dJ, dJ * TE1.size, TE1.size)

    max_TE1 = np.max(TE1)

    TE1 = TE1/max_TE1

    plt.plot(J, TE1, label=r'TE($\mathcal{B S}$('+str(driver)+r'$\rightarrow$'+str(target)+'))')

def TE2_normalised(target,driver,dJ):
    extension = str(target)+"_"+str(driver)
    filename = data_path+"\Data\BinaryTrees\BinaryTree_"
    
    filenameTE2 = filename+"TE2_"+extension

    TE2 = np.loadtxt(filenameTE2)

    J = np.linspace(dJ, dJ * TE2.size, TE2.size)


    max_TE2 = np.max(TE2)

    TE2 = TE2/max_TE2   



    # Plotting Chi, TE1, and TE2 vs J on the same plot
    #plt.plot(J, Chi, label='Chi('+str(driver)+','+str(target)+')')
    plt.plot(J, TE2, label=r'TE($\mathcal{B S}$('+str(driver)+r'$\rightarrow$'+str(target)+'))')

def TE1_(driver,target,dJ):
    extension = str(driver)+"_"+str(target)
    filename = data_path+"\Data\BinaryTrees\BinaryTree_"

    filenameTE1 = filename+"TE1_"+extension

    TE1 = np.loadtxt(filenameTE1)

    J = np.linspace(dJ, dJ * TE1.size, TE1.size)

    plt.plot(J, TE1, label=r'TE($\mathcal{B S}$('+str(driver)+r'$\rightarrow$'+str(target)+'))')


def TE2_(target,driver,dJ):
    extension = str(target)+"_"+str(driver)
    filename = data_path+"\Data\BinaryTrees\BinaryTree_"
    
    filenameTE2 = filename+"TE2_"+extension

    TE2 = np.loadtxt(filenameTE2)

    J = np.linspace(dJ, dJ * TE2.size, TE2.size)

    plt.plot(J, TE2, label=r'TE($\mathcal{B S}$('+str(driver)+r'$\rightarrow$'+str(target)+'))')



def check():
    maxes = [[0] * 6 for _ in range(6)]
    dJ = 0.015;
    for target in range(6):
        for driver in range(target+1):
            extension = str(driver)+"_"+str(target)
            filename = data_path+"\Data\BinaryTrees\BinaryTree_"
            filenameChi = filename+"Chi_"+extension
            filenameTE1 = filename+"TE1_"+extension
            filenameTE2 = filename+"TE2_"+extension

            Chi = np.loadtxt(filenameChi)
            TE1 = np.loadtxt(filenameTE1)
            TE2 = np.loadtxt(filenameTE2)

            J = np.linspace(dJ, dJ * TE1.size, TE1.size)
            
            maxes[driver][target]=fit_polynomial_and_find_max_x(J, TE1, 20)
            maxes[target][driver]=fit_polynomial_and_find_max_x(J, TE2, 20)

    

    max_group = [[0] * 3 for _ in range(3)]
    for a in [0,1,2]:
        for b in [0,1,2]:
            for j in [0,1]:
                for k in [0,1]:
                    driver = 2*a + j
                    target = 2*b + k
                    max_group[a][b] += maxes[driver][target]
    for a in [0,1,2]:
        for b in [0,1,2]:        
            for j in [0,1]:
                for k in [0,1]:
                    driver = 2*a + j
                    target = 2*b + k
                    maxes[driver][target] = round(max_group[a][b]/4,3)
    
    for row in range(6):
        string = "[ "+str(row)+" ], "
        for col in range(6):
            string += "[ "+str(maxes[row][col])+" ], "
        
        print(string)

# Create the plot outside the loop
plt.figure()

'''
# Loop over driver values from 0 to 6
for target in range(6):    
    driver = 0;
    plot_bin_star_normalised(driver,target, 0.015)

check();


check()


for a in [0,1,2]:
    for b in [0,1,2]: 
        for j in [0,1]:
            for k in [0,1]:
                driver = 2*a + j
                target = 2*b + k
                if(driver <= target):
                    TE1_(driver,target,0.015);
                else:
                    TE2_(target,driver,0.015);
        plt.show()
'''
#check()

for a in [0,1,2]:
    for b in [0,1,2]:
        for j in [0,1]:
            for k in [0,1]:
                driver = 2*a + j
                target = 2*b + k
                if(driver <= target):
                    TE1_(driver,target,0.015);
                else:
                    TE2_(target,driver,0.015);
        plt.xlabel(r'$\beta$')
        plt.ylabel('bits')
        plt.legend()
        plt.title('pairwise TE between centers of binary stars')
        plt.show()




plt.show()  
