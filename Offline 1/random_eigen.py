import numpy as np

def getRandomMatrix(n):
    while True:
        matrix = np.random.randint(1, 100, size=(n, n))
        if np.linalg.det(matrix) != 0:
            return matrix

n = int(input("Enter the dimension of the matrix: "))
A = getRandomMatrix(n)
eigenvalues, eigenvectors = np.linalg.eig(A)

reconA = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors) # A = X*lambda*I*X^-1
reconA = np.real(np.round(reconA, 0))

verdict = np.allclose(A, reconA)
print("A == Reconstructed A:", verdict)
print("\n")

print("A = \n", A)
print("Reconstructed A = \n", reconA)