##
# Main function of the Python program.
#
##

import pandas as pd
from comtypes.safearray import numpy as np
from numpy.core._multiarray_umath import matmul


# Fill in Those functions:

def computeRMSE(trueVector, EstimateVector):
    pass


def computeRadarJacobian(Xvector):
    pass


def computeCovMatrix(deltaT, sigma_aX, sigma_aY):
    pass


def computeFmatrix(state_vector, deltaT):
    """
       :param deltaT:
       :param state_vector
       :return: f_matrix
       """
    f_matrix = []
    for i in range(0, len(state_vector)):
        q = []
        for j in range(0, len(state_vector)):
            if i == j:
                q.append(1)
            elif j - i == len(state_vector) / 2:
                q.append(deltaT)
            else:
                q.append(0)
        f_matrix.append(q)
    return f_matrix


def main():
    my_cols = ["A", "B", "C", "D", "E", "f", "g", "h", "i", "j", "k"]
    data = pd.read_csv("./Root/src/data.txt", names=my_cols, delim_whitespace=True, header=None)
    print(data.head())
    for i in range(10):
        measur = data.iloc[i, :].values
        print(measur[0])
    # define matrices:
    deltaT = 0.1
    useRadar = False
    P = np.array
    xEstimate = []
    xTrue = []
    # H_Lidar =
    R_lidar = np.array([[0.0225, 0.0], [0.0, 0.0225]])

    R_radar = np.array([[0.9, 0, 0], [0.0, 0.0009, 0], [0, 0, 0.09]])

    F_matrix = computeFmatrix(xTrue, deltaT)
    X_state_current = []
    X_true_current = []
    firstMeasurment = data.iloc[0, :].values
    timeStamp = firstMeasurment[3]
    # fill in X_true and X_state. Put 0 for the velocities
    for index in range(1, len(data)):
        currentMeas = data.iloc[i, :].values

        # compute the current dela t
        if currentMeas[0] == 'L':
            deltaT = (currentMeas[3] - timeStamp) / 1000000
            timeStamp = currentMeas[3]

            # perfrom predict
            # X_state_current =
            # P =

            # pefrom measurment update
            # z =
            # S =
            # K =
            # X_state_current =
            # P =

        if currentMeas[0] == 'R' and useRadar:
            # perfrom predict
            deltaT = (currentMeas[4] - timeStamp) / 1000000
            timeStamp = currentMeas[4]
            # X_state_current =
            # P =

            # pefrom measurment update
            # jacobian =
            # z =
            # S =
            # K =
            # X_state_current =
            # P =

        xEstimate.append(X_state_current)
        xTrue.append(X_true_current)

    rmse = computeRMSE(xEstimate, xTrue)
    print(rmse)


if __name__ == '__main__':
    main()
