from turtle import color
from matplotlib import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Q3_1_ plot eigen vectors and compute covariance matrix
def plot_eigen_vectors(cords):
    b1 = cords[:,0]
    b2 = cords[:,1]
    X = np.column_stack([b1, b2])

    X_mean = np.mean(b1)
    Y_mean = np.mean(b2)
    X[:,0]=X[:,0]-X_mean
    X[:,1]=X[:,1]-Y_mean
    fact = X[:,0].shape[0]-1
    by_hand = np.dot(X.T, X.conj()) / fact

    print('\nCovariance Matrix Calcualtion Manually:\n','\n', by_hand)

    using_cov = np.cov(b1,b2)
    
    print('\n-------------Verification ------------\n')
    print('Covariance Martix Calcualtion Using Inbult function :\n','\n',using_cov,'\n')
   
    eigvals, eigvecs = np.linalg.eig(by_hand)

    print('Eigen Values : \n','\n',eigvals)
    print('\nEigen Vectors : \n','\n',eigvecs,'\n')
    origin = [X_mean,Y_mean]

    plt.title('Plot of Eigen Vectors')
    plt.scatter(cords[:,0],cords[:,1])
    plt.quiver(*origin,eigvecs[:,0],eigvecs[:,1],color='red',scale=10)
    plt.show()

##Q3_2_least_square_method
def plot_line_ls(cordinates):
    x_val = cordinates[:,0]
    y_val = cordinates[:,1]

    x_mean= np.mean(x_val)
    y_mean= np.mean(y_val)

    num = np.sum(((x_val[:]-x_mean)*(y_val[:]-y_mean)),dtype=np.float64)
    den = np.sum(((x_val[:] - x_mean)**2),dtype=np.float64)

    m = num / den

    c = y_mean - m*x_mean

    y_plot = m*x_val + c

    plt.title('Plot of Least Squares')
    plt.scatter(x_val,y_val)
    plt.plot([min(x_val), max(x_val)], [min(y_plot), max(y_plot)], color = 'red')
    plt.show()


##Q3_3_total_least_squares_method
def tls_coeffecients(cordinates):
    x = cordinates[:,0]
    y = cordinates[:,1]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_1 = x - x_mean
    y_1 = y - y_mean
    A_Matrix = np.vstack((x_1,y_1)).T

    B_Matrix = np.dot(A_Matrix.transpose(), A_Matrix)    

    K_Matrix = np.dot(B_Matrix.T,B_Matrix)
    w_vect, v_vect = np.linalg.eig(K_Matrix)
    index = np.argmin(w_vect)  
    coef = v_vect[:, index]
    a, b = coef
    c = (-1*a*x_mean) + (-1*b*y_mean)
    coefficents = np.array([a,b,c])
    return coefficents

def plot_line_tls(cordinates,coefficents):
    a,b,c = coefficents
    x = cordinates[:,0]
    y = cordinates[:,1]

    y_fitted = -1 *(a/b)*x + (-1*(c/b))

    plt.scatter(x,y)
    plt.title("Plot of Total Least Squares")
    plt.plot(x,y_fitted,color = "red")
    plt.show()


##Q3_3_ calcualting and ploting RANSAC
def ransac_error_calculation(points, coefficients):
    x = points[:,0]
    y = points[:,1]
    # x_sq = x ** 2

    a, b, c = coefficients

    E = np.square((a * x) + (b * y) + (c) )
    
    return E

def plot_line_ransac(cordinates,coefficient):
    a,b,c = coefficient
    x = cordinates[:,0]
    y = cordinates[:,1]

    y_fitted = -1 *(a/b)*x + (-1*(c/b))

    plt.scatter(x,y)
    plt.title("Plot of Ransac")
    plt.plot(x,y_fitted,color = "red")
    plt.show()

def ransac_coeffecients(points):
    outliers = 50
    accuracy = 0.95
    thresh = 10
    x = points[:,0]
    y = points[:,1]

    Number_points = points.shape[0]

    Best_selection_error = 0
    Best_Coefficient = np.zeros([3, 1])


    e = outliers / points.shape[0]
    s = 3
    p = accuracy
    ite = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
    ite = np.int(ite)
    ite = np.maximum(ite, 40)

    for i in range(ite):
        number_r = points.shape[0]
        random_points = np.random.choice(number_r, size=3)
        x_rand = x[random_points]
        y_rand = y[random_points]
        points_random = np.array([x_rand, y_rand]).T
        
        coeffecient_rand = tls_coeffecients(points_random)
        if np.any(np.iscomplex(coeffecient_rand)):
            continue
        Error = ransac_error_calculation(points, coeffecient_rand)
     
        for i in range(len(Error)):
            if float(Error[i]) > thresh:
                Error[i] = 0
            else:
                Error[i] = 1

        Current_Error = np.sum(Error)
        if Current_Error > Best_selection_error:
            Best_selection_error = Current_Error
            Best_Coefficient = coeffecient_rand
        
        if Best_selection_error/Number_points >= accuracy:
            break
    
    return Best_Coefficient 


def main():
    df = pd.read_csv(r'Resources/ENPM673_hw1_linear_regression_dataset - Sheet1.csv')
    cords = df[['age','charges']].to_numpy(dtype='int')
    coefficient = tls_coeffecients(cords)
    plot_line_ls(cords)
    plot_line_tls(cords,coefficient)
    r_coefficient=ransac_coeffecients(cords)
    plot_line_ransac(cords,r_coefficient)
    plot_eigen_vectors(cords)

if __name__ ==  '__main__':
    main()