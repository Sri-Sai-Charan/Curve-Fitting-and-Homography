import numpy as np

def main():
    A = np.array([[-5, -5, -1, 0, 0, 0, 500, 500, 100], 
                [0, 0, 0, -5, -5, -1, 500, 500, 100], 
                [-150, -5, -1, 0, 0, 0, 30000, 1000, 200], 
                [0, 0, 0, -150, -5, -1, 12000, 400, 80], 
                [-150, -150, -1, 0, 0, 0, 33000, 33000, 220], 
                [0, 0, 0, -150, -150, -1, 12000, 12000, 80], 
                [-5, -150, -1, 0, 0, 0, 500, 15000, 100], 
                [0, 0, 0, -5, -150, -1, 1000, 30000, 200]])

    m, n = A.shape

    AA_t = np.dot(A, A.transpose())
    A_tA = np.dot(A.transpose(), A) 

    eigen_values_1, U = np.linalg.eig(AA_t)
    eigen_values_2, V = np.linalg.eig(A_tA)


    #sort the eigen values and vectors
    index_1 = np.flip(np.argsort(eigen_values_1))
    eigen_values_1 = eigen_values_1[index_1]
    U = U[:, index_1]

    index_2 = np.flip(np.argsort(eigen_values_2))
    eigen_values_2 = eigen_values_2[index_2]
    V = V[:, index_2]

    E = np.zeros([m, n])

    var = np.minimum(m, n)

    for j in range(var):
        E[j,j] = np.abs(np.sqrt(eigen_values_1[j]))  

    Homography_Mat_ver = V[:, V.shape[1] - 1]
    Homography = Homography_Mat_ver.reshape([3,3])
    Homography = Homography / Homography[2,2]
    print("\nMatrix given : \n")
    print(A)
    print("\nHomography matrix  : \n")
    print(Homography)

if __name__=='__main__':
    main()