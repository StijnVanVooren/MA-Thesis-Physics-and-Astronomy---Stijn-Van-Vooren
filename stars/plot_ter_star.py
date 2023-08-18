from os.path import dirname, abspath
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import warnings

from sklearn.metrics import r2_score
from scipy.stats import beta

from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore', 'Polyfit may be poorly conditioned')

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



def TE01_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_01_"+extension

    TE01 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE01.size, TE01.size)

    plt.plot(J, TE01, label=r'TE$_{01}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE02_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_02_"+extension

    TE02 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE02.size, TE02.size)

    plt.plot(J, TE02, label=r'TE$_{02}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE10_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_10_"+extension

    TE10 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE10.size, TE10.size)

    plt.plot(J, TE10, label=r'TE$_{10}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE12_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_12_"+extension

    TE12 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE12.size, TE12.size)

    plt.plot(J, TE12, label=r'TE$_{12}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE20_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_20_"+extension

    TE20 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE20.size, TE20.size)

    plt.plot(J, TE20, label=r'TE$_{20}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE21_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_21_"+extension

    TE21 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE21.size, TE21.size)

    plt.plot(J, TE21, label=r'TE$_{21}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE0_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_0_"+extension

    TE0 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE0.size, TE0.size)

    plt.plot(J, TE0, label=r'TE$_{0}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE1_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_1_"+extension

    TE1 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE1.size, TE1.size)

    plt.plot(J, TE1, label=r'TE$_{1}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def TE2_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE = filename+"TE_2_"+extension

    TE2 = np.loadtxt(filenameTE)

    J = np.linspace(dJ, dJ * TE2.size, TE2.size)

    plt.plot(J, TE2, label=r'TE$_{2}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def ST0_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE1 = filename+"TE_10_"+extension
    filenameTE2 = filename+"TE_20_"+extension

    TE1 = np.loadtxt(filenameTE1)
    TE2 = np.loadtxt(filenameTE2)

    filenameTE = filename+"TE_0_"+extension

    TE = np.loadtxt(filenameTE)

    red = np.minimum(TE1,TE2)

    syn = TE - np.maximum(TE1,TE2)

    J = np.linspace(dJ, dJ * TE2.size, TE2.size)

    

    plt.plot(J, TE2, label=r'Syn$_{0}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def ST1_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE1 = filename+"TE_01_"+extension
    filenameTE2 = filename+"TE_21_"+extension

    TE1 = np.loadtxt(filenameTE1)
    TE2 = np.loadtxt(filenameTE2)

    filenameTE = filename+"TE_1_"+extension

    TE = np.loadtxt(filenameTE)

    red = np.minimum(TE1,TE2)

    syn = TE - np.maximum(TE1,TE2)

    J = np.linspace(dJ, dJ * TE2.size, TE2.size)

    

    plt.plot(J, TE2, label=r'Syn$_{1}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')

def ST2_(c1,c2,c3,dJ):
    extension = str(c1)+"_"+str(c2)+"_"+str(c3)
    filename = data_path+"\Data\TernaryStars\TernaryTree_"

    filenameTE1 = filename+"TE_02_"+extension
    filenameTE2 = filename+"TE_12_"+extension

    TE1 = np.loadtxt(filenameTE1)
    TE2 = np.loadtxt(filenameTE2)

    filenameTE = filename+"TE_2_"+extension

    TE = np.loadtxt(filenameTE)

    red = np.minimum(TE1,TE2)

    syn = TE - np.maximum(TE1,TE2)

    J = np.linspace(dJ, dJ * TE2.size, TE2.size)

    

    plt.plot(J, TE2, label=r'Syn$_{2}(\mathcal{T S}$('+str(c1)+','+str(c2)+','+str(c3)+'))')




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


def find_max_fit(X, Y, degree=20,resolution=1000):
    # Find the X value corresponding to the maximum Y value
    max_index = np.argmax(Y)
    max_X = X[max_index]
    
    # Define a region around the maximum to perform the polynomial fit
    fit_range = np.abs(X - max_X) < 0.2  # Adjust the value (0.1) to your desired fit range
    
    # Perform polynomial fit
    coefficients = np.polyfit(X[fit_range], Y[fit_range], degree)
    
    # Create a polynomial function based on the coefficients
    polynomial = np.poly1d(coefficients)
    
    # Create a higher-resolution X array within the specified range
    fit_X_values = np.linspace(X[fit_range].min(), X[fit_range].max(), resolution)

    # Calculate the corresponding Y values using the fitted polynomial
    fit_Y_values = polynomial(fit_X_values)

    # Find the maximum of the fit
    max_fit_index = np.argmax(fit_Y_values)
    max_fit_X = fit_X_values[max_fit_index]
    max_fit_Y = fit_Y_values[max_fit_index]

    # Plot the data, fit, and maximum of the fit
    '''
    plt.plot(X, Y, 'o', label='Data')
    plt.plot(fit_X_values, fit_Y_values, label='Fitted Curve')
    plt.plot(max_fit_X, max_fit_Y, 'ro', label='Maximum of Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    
    return max_fit_X, max_fit_Y

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



'''
for a in [0,1,2]:
    for b in [0,1,2]:
        for c in range(15)
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



for a in range(8):
    for b in range(8):
        for c in range(8):
            for i in [0,1]:
                for j in [0,1]:
                    for k in [0,1]:
                        c1 = 2*a+i
                        c2 = 2*b+j+1
                        c3 = 2*c+k
                        if(c1<=c3):
                            try:
                                ST0_(c1,c2,c3,0.005)
                            except FileNotFoundError:
                                print(f"The file is not found. Skipping the for loop.")
                            
                            try:
                                ST0_(c1,c2,c3,0.005)
                            except FileNotFoundError:
                                print(f"The file is not found. Skipping the for loop.")
                            try:
                                ST2_(c1,c2,c3,0.005)
                            except FileNotFoundError:
                                print(f"The file is not found. Skipping the for loop.")
                                
                            
                        else:
                            try:
                                ST2_(c1,c2,c3,0.005)
                            except FileNotFoundError:
                                print(f"The file is not found. Skipping the for loop.")
                            
                            try:
                                ST1_(c3,c2,c1,0.005)
                            except FileNotFoundError:
                                print(f"The file is not found. Skipping the for loop.")
                            try:
                                ST2_(c3,c2,c1,0.005)
                            except FileNotFoundError:
                                print(f"The file is not found. Skipping the for loop.")
                            
            plt.xlabel(r'$\beta$')
            plt.ylabel('bits')
            plt.legend()
            plt.title('information measures between centers of ternary stars')
            plt.show()

'''
filename = data_path+"\Data\TernaryStars\TernaryTree_"
dJ = 0.005


for a in range(11):
    for b in range(11):
        for c in range(11):
            N = 0
            for i in [0,1]:
                for j in [0,1]:
                    for k in [0,1]:
                        c1 = 2*a+i
                        c2 = 2*b+j-1
                        c3 = 2*c+k
                        
                        if c2 == -1:
                            continue
                        
                        extension = str(c1)+"_"+str(c2)+"_"+str(c3)

                        B_TE01 = 0
                        B_TE02 = 0
                        B_TE10 = 0
                        B_TE12 = 0
                        B_TE20 = 0
                        B_TE21 = 0

                        B_TE0 = 0
                        B_TE1 = 0
                        B_TE2 = 0

                        B_ST0 = 0
                        B_ST1 = 0
                        B_ST2 = 0

                        V_TE01 = 0
                        V_TE02 = 0
                        V_TE10 = 0
                        V_TE12 = 0
                        V_TE20 = 0
                        V_TE21 = 0

                        V_TE0 = 0
                        V_TE1 = 0
                        V_TE2 = 0

                        V_ST0 = 0
                        V_ST1 = 0
                        V_ST2 = 0

                        N = 0                    

                        if(c1>c3):
                            continue
                        
                        try:
                            filenameTE = filename+"TE_01_"+extension
                            TE01 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE01.size, TE01.size)
                            maxX_01, maxY_01 = find_max_fit(J,TE01)

                            filenameTE = filename+"TE_02_"+extension
                            TE02 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE02.size, TE02.size)
                            maxX_02, maxY_02 = find_max_fit(J,TE02)

                            filenameTE = filename+"TE_10_"+extension
                            TE10 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE10.size, TE10.size)
                            maxX_10, maxY_10 = find_max_fit(J,TE10)

                            filenameTE = filename+"TE_12_"+extension
                            TE12 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE12.size, TE12.size)
                            maxX_12, maxY_12 = find_max_fit(J,TE12)

                            filenameTE = filename+"TE_20_"+extension
                            TE20 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE20.size, TE20.size)
                            maxX_20, maxY_20 = find_max_fit(J,TE20)

                            filenameTE = filename+"TE_21_"+extension
                            TE21 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE21.size, TE21.size)
                            maxX_21, maxY_21 = find_max_fit(J,TE21)
                        
                            filenameTE = filename+"TE_0_"+extension
                            TE0 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE0.size, TE0.size)
                            maxX_0, maxY_0 = find_max_fit(J,TE0)

                            filenameTE = filename+"TE_1_"+extension
                            TE1 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE1.size, TE1.size)
                            maxX_1, maxY_1 = find_max_fit(J,TE1)

                            filenameTE = filename+"TE_2_"+extension
                            TE2 = np.loadtxt(filenameTE)
                            J = np.linspace(0.005, 0.005 * TE2.size, TE2.size)
                            maxX_2, maxY_2 = find_max_fit(J,TE2)

                            #ST0
                            filenameTE1 = filename+"TE_10_"+extension
                            filenameTE2 = filename+"TE_20_"+extension

                            TE1 = np.loadtxt(filenameTE1)
                            TE2 = np.loadtxt(filenameTE2)

                            filenameTE = filename+"TE_0_"+extension

                            TE = np.loadtxt(filenameTE)                           

                            syn = TE - np.maximum(TE1,TE2)

                            J = np.linspace(dJ, dJ * TE2.size, TE2.size)
                            maxX_ST_0, maxY_ST_0 = find_max_fit(J,syn)

                            #ST1
                            filenameTE1 = filename+"TE_01_"+extension
                            filenameTE2 = filename+"TE_21_"+extension

                            TE1 = np.loadtxt(filenameTE1)
                            TE2 = np.loadtxt(filenameTE2)

                            filenameTE = filename+"TE_1_"+extension

                            TE = np.loadtxt(filenameTE)                           

                            syn = TE - np.maximum(TE1,TE2)

                            J = np.linspace(dJ, dJ * TE2.size, TE2.size)
                            maxX_ST_1, maxY_ST_1 = find_max_fit(J,syn)

                            #ST2
                            filenameTE1 = filename+"TE_02_"+extension
                            filenameTE2 = filename+"TE_12_"+extension

                            TE1 = np.loadtxt(filenameTE1)
                            TE2 = np.loadtxt(filenameTE2)

                            filenameTE = filename+"TE_2_"+extension

                            TE = np.loadtxt(filenameTE)                           

                            syn = TE - np.maximum(TE1,TE2)

                            J = np.linspace(dJ, dJ * TE2.size, TE2.size)
                            maxX_ST_2, maxY_ST_2 = find_max_fit(J,syn)
                            

                            N = N + 1

                            B_TE01 += maxX_01
                            B_TE02 += maxX_02
                            B_TE10 += maxX_10
                            B_TE12 += maxX_12
                            B_TE20 += maxX_20
                            B_TE21 += maxX_21

                            B_TE0 += maxX_0
                            B_TE1 += maxX_1
                            B_TE2 += maxX_2

                            B_ST0 += maxX_ST_0
                            B_ST1 += maxX_ST_1
                            B_ST2 += maxX_ST_2

                            V_TE01 += maxY_01
                            V_TE02 += maxY_02
                            V_TE10 += maxY_10
                            V_TE12 += maxY_12
                            V_TE20 += maxY_20
                            V_TE21 += maxY_21

                            V_TE0 += maxY_0
                            V_TE1 += maxY_1
                            V_TE2 += maxY_2

                            V_ST0 += maxY_ST_0
                            V_ST1 += maxY_ST_1
                            V_ST2 += maxY_ST_2
                            
                        except FileNotFoundError:
                            #print(str(a)+","+str(b)+","+str(c))
                            continue
                                #print(f"The file is not found. Skipping the for loop.")
            if N != 0:
                B_TE01 = B_TE01/N
                B_TE02 = B_TE02/N
                B_TE10 = B_TE10/N
                B_TE12 = B_TE12/N
                B_TE20 = B_TE20/N
                B_TE21 = B_TE21/N

                B_TE0 = B_TE0/N
                B_TE1 = B_TE1/N
                B_TE2 = B_TE2/N

                B_ST0 = B_ST0/N
                B_ST1 = B_ST1/N
                B_ST2 = B_ST2/N

                V_TE01 = V_TE01/N
                V_TE02 = V_TE02/N
                V_TE10 = V_TE10/N
                V_TE12 = V_TE12/N
                V_TE20 = V_TE20/N
                V_TE21 = V_TE21/N
                
                V_TE0 = V_TE0/N
                V_TE1 = V_TE1/N
                V_TE2 = V_TE2/N

                V_ST0 = V_ST0/N
                V_ST1 = V_ST1/N
                V_ST2 = V_ST2/N
            if B_TE01 != 0 and B_TE02!=0 and B_TE10 != 0 and B_TE12 != 0 and B_TE20 != 0 and B_TE21 != 0 and B_TE0 != 0 and B_TE1 != 0 and B_TE2 != 0 and B_ST0 != 0 and B_ST1 != 0 and B_ST2 != 0:

                if a == c:
                    B_TE01 = np.mean([B_TE01,B_TE21])
                    B_TE21 = B_TE01
                    B_TE10 = np.mean([B_TE10,B_TE12])
                    B_TE12 = B_TE10
                    B_TE20 = np.mean([B_TE20,B_TE02])
                    B_TE02 = B_TE20

                    B_TE0 = np.mean([B_TE0,B_TE2])
                    B_TE2 = B_TE0

                    B_ST0 = np.mean([B_ST0,B_ST2])
                    B_ST2 = B_ST0
                if a==b:
                    B_TE01 = np.mean([B_TE01,B_TE10])
                    B_TE10 = B_TE01                   
                    
                if b==c:                                        
                    B_TE21 = np.mean([B_TE21,B_TE12])
                    B_TE12 = B_TE21                     

                
                    
                string_to_print = "[ *$("+str(a)+","+str(b)+","+str(c)+")$* ], "

                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE02,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE20,2)))+" ], "                
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE21,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE01,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE10,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE12,2)))+" ], "
                
                

                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE0,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE1,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_TE2,2)))+" ], "

                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_ST0,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_ST1,2)))+" ], "
                string_to_print = string_to_print + "[ " + str("{:.2f}".format(round(B_ST2,2)))+" ], "

                print(string_to_print)
                        
                                
                                
                            
                          
'''
plt.xlabel(r'$\beta$')
plt.ylabel('bits')
plt.legend()
plt.title('information measures between centers of ternary stars')
plt.show()
'''
