# kurtis bertauche
# cs 455
# project 1

from math import sqrt, cos, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from celluloid import Camera
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\kurti\\Downloads\\ffmpeg-6.1.1-essentials_build\\ffmpeg-6.1.1-essentials_build\\bin\\ffmpeg.exe'

C_1_ALPHA = 30
C_2_ALPHA = 2 * sqrt(C_1_ALPHA)
C_MT_1 = 1.1
C_MT_2 = 2 * sqrt(C_MT_1)
Q_MT = (150, 150)
EPSILON = 0.1
H = 0.2
A = B = 5
C = (abs(A - B) / sqrt(4 * A * B))
NUM_NODES = 100
DIMENSIONS = 2
D = 15
K = 1.2
R = K * D
DELTA_T = 0.009
RANDOM_SEED = 17
np.random.seed(17)

def main():

    ################################################################
    # CASE 1 - ALGORITHM 1 - FRAGMENTATION
    ################################################################

    # initialize nodes
    # nodePositions = np.random.randint(low = 0, high = 50, size = (DIMENSIONS, NUM_NODES))
    # nodeVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    # nodeAccelerations = np.zeros((DIMENSIONS, NUM_NODES))
    # allPositions = nodePositions.copy().reshape(DIMENSIONS, NUM_NODES, 1)
    # allNodeVelocities = nodeVelocities.copy().reshape(DIMENSIONS, NUM_NODES, 1)

    # plotNodesAndSave(nodePositions, "scatter/scatter_start.png")

    # for i in range(0, 500):
    #     nodePositions, nodeVelocities, nodeAccelerations = updateAlgo1(nodePositions, nodeVelocities)
    #     if (i + 1) % 100 == 0:
    #         plotNodesAndSave(nodePositions, "scatter/scatter" + str(i + 1) + ".png")
    #     print("FINISHED ITERATION: ", i)
    #     allPositions = np.concatenate((allPositions, nodePositions.copy()[:, :, np.newaxis]), axis = 2)
    #     allNodeVelocities = np.concatenate((allNodeVelocities, nodeVelocities.copy()[:, :, np.newaxis]), axis = 2)

    # plotNodesAndSave(nodePositions, "scatter/scatter_end.png")
    # plotAndSaveNodeTrajectories(allPositions, "scatter/scatter_trajectory.png")
    # plotAndSaveNodeVelocities(allNodeVelocities, "scatter/scatter_individual_velocity.png", "scatter/scatter_all_velcoity.png")
    # plotAndSaveConnectivity(allPositions, "scatter/scatter_connectivity.png")

    ################################################################
    # CASE 2 - ALGORITHM 2 - STATIC TARGET
    ################################################################

    # nodePositions = np.random.randint(low = 0, high = 50, size = (DIMENSIONS, NUM_NODES))
    # nodeVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    # nodeAccelerations = np.zeros((DIMENSIONS, NUM_NODES))
    # allPositions = nodePositions.copy().reshape(DIMENSIONS, NUM_NODES, 1)
    # allNodeVelocities = nodeVelocities.copy().reshape(DIMENSIONS, NUM_NODES, 1)

    # plotNodesAndSave(nodePositions, "static/static_start.png")

    # for i in range(0, 500):
    #     nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2(nodePositions, nodeVelocities)
    #     print("FINISHED ITERATION: ", i)
    #     if (i + 1) % 100 == 0:
    #         plt.clf()
    #         plotNodes(nodePositions)
    #         plt.scatter(Q_MT[0], Q_MT[1], color = "red")
    #         plt.savefig("static/static" + str(i + 1) + ".png")
    #     allPositions = np.concatenate((allPositions, nodePositions.copy()[:, :, np.newaxis]), axis = 2)
    #     allNodeVelocities = np.concatenate((allNodeVelocities, nodeVelocities.copy()[:, :, np.newaxis]), axis = 2)

    # plotNodesAndSave(nodePositions, "static/static_end.png")
    # plotAndSaveNodeTrajectories(allPositions, "static/static_trajectory.png")
    # plotAndSaveNodeVelocities(allNodeVelocities, "static/static_individual_velocity.png", "static/static_all_velcoity.png")
    # plotAndSaveConnectivity(allPositions, "static/static_connectivity.png")

    ################################################################
    # CASE 3 - ALGORITHM 2 - SIN WAVE
    ################################################################

    nodePositions = np.random.randint(low = 0, high = 150, size = (DIMENSIONS, NUM_NODES))
    nodeVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    nodeAccelerations = np.zeros((DIMENSIONS, NUM_NODES))
    allPositions = nodePositions.copy().reshape(DIMENSIONS, NUM_NODES, 1)
    allNodeVelocities = nodeVelocities.copy().reshape(DIMENSIONS, NUM_NODES, 1)
    allXCenters = []
    allYCenters = []
    gammaX = np.arange(500)
    gammaY = (np.sin(2 * np.pi * gammaX / 500) * 75) + 75

    for i in range(0, 500):
        if i == 0:
            nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), (0, 0))
        else:
            nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), ((gammaX[i] - gammaX[i - 1])/DELTA_T, (gammaY[i] - gammaY[i - 1])/DELTA_T))
            
        allPositions = np.concatenate((allPositions, nodePositions.copy()[:, :, np.newaxis]), axis = 2)
        allNodeVelocities = np.concatenate((allNodeVelocities, nodeVelocities.copy()[:, :, np.newaxis]), axis = 2)

        thisXCenter = nodePositions[0, :].copy()
        thisXCenter = np.sum(thisXCenter)
        thisXCenter /= NUM_NODES
        allXCenters.append(thisXCenter)
        thisYCenter = nodePositions[1, :].copy()
        thisYCenter = np.sum(thisYCenter)
        thisYCenter /= NUM_NODES
        allYCenters.append(thisYCenter)
            

        if ((i + 1) % 100 == 0) or (i == 0):
            plt.clf()
            plt.plot(gammaX, gammaY, color = "red", label = "Gamma Agent")
            plt.plot(allXCenters, allYCenters, color = "black", label = "Center of Mass")
            plotNodes(nodePositions)
            plt.scatter(gammaX[i], gammaY[i], color = "green", zorder = 100, label = "Target")
            plt.legend(bbox_to_anchor = (1, -0.3), borderaxespad = 0)
            plt.title("Case 3: Sin Wave")
            plt.savefig("sin/sin" + str(i + 1) + ".png")
        print("FINISHED ITERATION: ", i)

    plotAndSaveNodeTrajectories(allPositions, "sin/sin_trajectory.png")
    plotAndSaveNodeVelocities(allNodeVelocities, "sin/sin_individual_velocity.png", "sin/sin_all_velcoity.png")
    plotAndSaveConnectivity(allPositions, "sin/sin_connectivity.png")
    plotCenterOfMassAndTarget(gammaX, gammaY, allPositions, "sin/center_of_mass.png")

    ################################################################
    # CASE 4 - ALGORITHM 2 - CIRCLE
    ################################################################

    nodePositions = np.random.randint(low = 0, high = 150, size = (DIMENSIONS, NUM_NODES))
    nodeVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    nodeAccelerations = np.zeros((DIMENSIONS, NUM_NODES))
    allPositions = nodePositions.copy().reshape(DIMENSIONS, NUM_NODES, 1)
    allNodeVelocities = nodeVelocities.copy().reshape(DIMENSIONS, NUM_NODES, 1)
    thetaOne = np.linspace(1 * np.pi, 0, 250)
    thetaTwo = np.linspace(2 * np.pi, 1 * np.pi, 250)
    theta = np.append(thetaOne, thetaTwo)
    radius = 150
    gammaX = radius * np.cos(theta)
    gammaX = gammaX + 250
    gammaY = radius * np.sin(theta)
    gammaY = gammaY + 150
    allXCenters = []
    allYCenters = []

    for i in range(0, 500):
        if i == 0:
            nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), (0, 0))
        else:
            nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), ((gammaX[i] - gammaX[i - 1])/DELTA_T, (gammaY[i] - gammaY[i - 1])/DELTA_T))
            
        allPositions = np.concatenate((allPositions, nodePositions.copy()[:, :, np.newaxis]), axis = 2) 
        allNodeVelocities = np.concatenate((allNodeVelocities, nodeVelocities.copy()[:, :, np.newaxis]), axis = 2)

        thisXCenter = nodePositions[0, :].copy()
        thisXCenter = np.sum(thisXCenter)
        thisXCenter /= NUM_NODES
        allXCenters.append(thisXCenter)
        thisYCenter = nodePositions[1, :].copy()
        thisYCenter = np.sum(thisYCenter)
        thisYCenter /= NUM_NODES
        allYCenters.append(thisYCenter)

        if ((i + 1) % 100 == 0) or (i == 0):
            plt.clf()
            plt.plot(gammaX, gammaY, color = "red", label = "Gamma Agent")
            plt.plot(allXCenters, allYCenters, color = "black", label = "Center of Mass")
            plotNodes(nodePositions)
            plt.scatter(gammaX[i], gammaY[i], color = "green", zorder = 100, label = "Target")
            plt.legend(bbox_to_anchor = (0.15, 0.1), loc = "upper right")
            plt.title("Case 4: Circle")
            plt.savefig("circle/circle" + str(i + 1) + ".png")
        print("FINISHED ITERATION: ", i)

    plotAndSaveNodeTrajectories(allPositions, "circle/circle_trajectory.png")
    plotAndSaveNodeVelocities(allNodeVelocities, "circle/circle_individual_velocity.png", "circle/circle_all_velcoity.png")
    plotAndSaveConnectivity(allPositions, "circle/circle_connectivity.png")
    plotCenterOfMassAndTarget(gammaX, gammaY, allPositions, "circle/center_of_mass.png")



def updateAlgo1(nodePositions, nodeVelocities):

    newAcclerations = np.zeros((DIMENSIONS, NUM_NODES))
    newVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    newPositions = np.zeros((DIMENSIONS, NUM_NODES))
    # we must update the acceleration for every node
    for i in range(0, NUM_NODES):

        sumOfFirstPart = (0, 0)
        sumOfSecondPart = (0, 0)
        q_i = (nodePositions[0][i], nodePositions[1][i])
        p_i = (nodeVelocities[0][i], nodeVelocities[1][i])
        for j in range(0, NUM_NODES):
            if i == j:
                continue

            thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][j], nodePositions[1][j])))
            if thisPairNorm > R:
                continue

            q_j = (nodePositions[0][j], nodePositions[1][j])
            p_j = (nodeVelocities[0][j], nodeVelocities[1][j])
            # this is a neighbor node, need to calculate its effect for the new acceleration
            # first part
            this_sigma_norm = sigma_norm(tuple_diff((q_j[0], q_j[1]), (q_i[0], q_i[1])))
            this_phi_alpha = phi_alpha(this_sigma_norm)
            this_n_i_j = n_i_j(q_j, q_i)
            this_part = tuple_by_scalar(this_n_i_j, this_phi_alpha)
            sumOfFirstPart = tuple_sum(sumOfFirstPart, this_part)
            # seoncd part
            this_a_i_j = a_i_j(q_j, q_i)
            velocity_diff = tuple_diff(p_j, p_i)
            secondPart = tuple_by_scalar(velocity_diff, this_a_i_j)
            sumOfSecondPart = tuple_sum(sumOfSecondPart, secondPart)

        sumOfFirstPart = tuple_by_scalar(sumOfFirstPart, C_1_ALPHA)
        sumOfSecondPart = tuple_by_scalar(sumOfSecondPart, C_2_ALPHA)
        newAccel = tuple_sum(sumOfFirstPart, sumOfSecondPart)
        newAcclerations[0][i] = newAccel[0] # x
        newAcclerations[1][i] = newAccel[1] # y
        newVelocities[0][i] = nodeVelocities[0][i] + (newAccel[0] * DELTA_T) # x
        newVelocities[1][i] = nodeVelocities[1][i] + (newAccel[1] * DELTA_T) # y
        newPositions[0][i] = nodePositions[0][i] + (nodeVelocities[0][i] * DELTA_T) + ((1/2) * newAccel[0] * (DELTA_T * DELTA_T))
        newPositions[1][i] = nodePositions[1][i] + (nodeVelocities[1][i] * DELTA_T) + ((1/2) * newAccel[1] * (DELTA_T * DELTA_T))
    return newPositions, newVelocities, newAcclerations

def updateAlgo2(nodePositions, nodeVelocities):

    newAcclerations = np.zeros((DIMENSIONS, NUM_NODES))
    newVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    newPositions = np.zeros((DIMENSIONS, NUM_NODES))
    # we must update the acceleration for every node
    for i in range(0, NUM_NODES):

        sumOfFirstPart = (0, 0)
        sumOfSecondPart = (0, 0)
        q_i = (nodePositions[0][i], nodePositions[1][i])
        p_i = (nodeVelocities[0][i], nodeVelocities[1][i])
        for j in range(0, NUM_NODES):
            if i == j:
                continue
            thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][j], nodePositions[1][j])))
            if thisPairNorm > R:
                continue

            q_j = (nodePositions[0][j], nodePositions[1][j])
            p_j = (nodeVelocities[0][j], nodeVelocities[1][j])
            # this is a neighbor node, need to calculate its effect for the new acceleration
            # first part
            this_sigma_norm = sigma_norm(tuple_diff((q_j[0], q_j[1]), (q_i[0], q_i[1])))
            this_phi_alpha = phi_alpha(this_sigma_norm)
            this_n_i_j = n_i_j(q_j, q_i)
            this_part = tuple_by_scalar(this_n_i_j, this_phi_alpha)
            sumOfFirstPart = tuple_sum(sumOfFirstPart, this_part)
            # seoncd part
            this_a_i_j = a_i_j(q_j, q_i)
            velocity_diff = tuple_diff(p_j, p_i)
            secondPart = tuple_by_scalar(velocity_diff, this_a_i_j)
            sumOfSecondPart = tuple_sum(sumOfSecondPart, secondPart)

        sumOfFirstPart = tuple_by_scalar(sumOfFirstPart, C_1_ALPHA)
        sumOfSecondPart = tuple_by_scalar(sumOfSecondPart, C_2_ALPHA)
        staticTarget = tuple_diff(q_i, Q_MT)
        staticTarget = tuple_by_scalar(staticTarget, C_MT_1)
        newAccel = tuple_sum(sumOfFirstPart, sumOfSecondPart)
        newAccel = tuple_diff(newAccel, staticTarget)
        newAcclerations[0][i] = newAccel[0] # x
        newAcclerations[1][i] = newAccel[1] # y
        newVelocities[0][i] = nodeVelocities[0][i] + (newAccel[0] * DELTA_T) # x
        newVelocities[1][i] = nodeVelocities[1][i] + (newAccel[1] * DELTA_T) # y
        newPositions[0][i] = nodePositions[0][i] + (nodeVelocities[0][i] * DELTA_T) + ((1/2) * newAccel[0] * (DELTA_T * DELTA_T))
        newPositions[1][i] = nodePositions[1][i] + (nodeVelocities[1][i] * DELTA_T) + ((1/2) * newAccel[1] * (DELTA_T * DELTA_T))

    return newPositions, newVelocities, newAcclerations

def updateAlgo2DynamicTarget(nodePositions, nodeVelocities, gammaPos, gammaVelocity):

    newAcclerations = np.zeros((DIMENSIONS, NUM_NODES))
    newVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    newPositions = np.zeros((DIMENSIONS, NUM_NODES))
    # we must update the acceleration for every node
    for i in range(0, NUM_NODES):

        sumOfFirstPart = (0, 0)
        sumOfSecondPart = (0, 0)
        q_i = (nodePositions[0][i], nodePositions[1][i])
        p_i = (nodeVelocities[0][i], nodeVelocities[1][i])
        for j in range(0, NUM_NODES):
            if i == j:
                continue

            thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][j], nodePositions[1][j])))
            if thisPairNorm > R:
                continue

            q_j = (nodePositions[0][j], nodePositions[1][j])
            p_j = (nodeVelocities[0][j], nodeVelocities[1][j])
            # this is a neighbor node, need to calculate its effect for the new acceleration
            # first part
            this_sigma_norm = sigma_norm(tuple_diff((q_j[0], q_j[1]), (q_i[0], q_i[1])))
            this_phi_alpha = phi_alpha(this_sigma_norm)
            this_n_i_j = n_i_j(q_j, q_i)
            this_part = tuple_by_scalar(this_n_i_j, this_phi_alpha)
            sumOfFirstPart = tuple_sum(sumOfFirstPart, this_part)
            # seoncd part
            this_a_i_j = a_i_j(q_j, q_i)
            velocity_diff = tuple_diff(p_j, p_i)
            secondPart = tuple_by_scalar(velocity_diff, this_a_i_j)
            sumOfSecondPart = tuple_sum(sumOfSecondPart, secondPart)

        sumOfFirstPart = tuple_by_scalar(sumOfFirstPart, C_1_ALPHA)
        sumOfSecondPart = tuple_by_scalar(sumOfSecondPart, C_2_ALPHA)

        # gamma pos
        staticTarget = tuple_diff(q_i, gammaPos)
        staticTarget = tuple_by_scalar(staticTarget, C_MT_1)
        newAccel = tuple_sum(sumOfFirstPart, sumOfSecondPart)
        newAccel = tuple_diff(newAccel, staticTarget)

        # gamma velocity
        gammaVel = tuple_diff(p_i, gammaVelocity)
        gammaVel = tuple_by_scalar(gammaVel, C_MT_2)
        newAccel = tuple_diff(newAccel, gammaVel)

        newAcclerations[0][i] = newAccel[0] # x
        newAcclerations[1][i] = newAccel[1] # y
        newVelocities[0][i] = nodeVelocities[0][i] + (newAccel[0] * DELTA_T) # x
        newVelocities[1][i] = nodeVelocities[1][i] + (newAccel[1] * DELTA_T) # y
        newPositions[0][i] = nodePositions[0][i] + (nodeVelocities[0][i] * DELTA_T) + ((1/2) * newAccel[0] * (DELTA_T * DELTA_T))
        newPositions[1][i] = nodePositions[1][i] + (nodeVelocities[1][i] * DELTA_T) + ((1/2) * newAccel[1] * (DELTA_T * DELTA_T))

    return newPositions, newVelocities, newAcclerations

"""
Input: Vector (x, y)
Output: Euclidean Norm: Scalar
"""
def euclidean_norm(vector):
    squaredSum = 0
    for component in vector:
        squaredSum += (component * component)
    return sqrt(squaredSum)

"""
Input: Vector (x, y)
Output: Sigma Norm: Scalar
"""
def sigma_norm(vector):
    euclidean = euclidean_norm(vector)
    squared_euclidean = euclidean * euclidean
    return (1 / EPSILON) * (sqrt(1 + (EPSILON * squared_euclidean)) - 1)

"""
Input: Two vectors (x, y), (x, y)
Output: One vector (x, y)
"""
def n_i_j(vector1, vector2):
    difference = tuple_diff(vector1, vector2)
    euclidean = euclidean_norm(difference)
    squared_euclidean = euclidean * euclidean
    #return((difference) / (sqrt(1 + (EPSILON * squared_euclidean))))

    return tuple_div_by_scalar(difference, (sqrt(1 + (EPSILON * squared_euclidean))))

"""
Input: Two vectors (x, y), (x, y)
Output: One vector (x, y)
"""
def tuple_diff(vector1, vector2):
    return tuple(np.subtract(vector1, vector2))

def tuple_sum(vector1, vector2):
    return tuple(np.add(vector1, vector2))

def tuple_by_scalar(vector, scalar):
    return tuple([scalar * component for component in vector])

def tuple_div_by_scalar(vector, scalar):
    return tuple([component / scalar for  component in vector])

"""
Input: One scalar value, z
Output: One scalar value, z
"""
def rho_h(z):
    if z >= 0 and z < H:
        return 1
    elif z >= H and z <= 1:
        return ((1 / 2) * (1 + cos(pi * ((z - H)/(1 - H)))))
    else:
        return 0
    
R_ALPHA = euclidean_norm((R,))
D_ALPHA = euclidean_norm((D,))

"""
Input: Two vectors (x, y), (x, y)
Output: One scalar value, z
"""
def a_i_j(vector1, vector2):
    difference = tuple_diff(vector1, vector2)
    sigma_of_diff = sigma_norm(difference)
    rho = rho_h(sigma_of_diff / R_ALPHA)
    return rho

"""
Input: One scalar value, z
Output: One scalar value, z
"""
def phi_alpha(z):
    this_rho_h = rho_h(z / R_ALPHA)
    this_phi = phi(z - D_ALPHA)
    return this_rho_h * this_phi

"""
Input: One scalar value, z
Output: One scalar value, z
"""
def phi(z):
    bracket_portion = ((A + B) * sigma_one(z + C)) + (A - B)
    return ((1 / 2) * bracket_portion)

"""
Input: One scalar value, z
Output: One scalar value, z
"""
def sigma_one(z):
    return (z / (sqrt(1 + (z * z))))
    
def plotNodesAndSave(nodePositions, fileName):

    plt.clf()
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):

            # nodes should not be used with themselves
            if i == j:
                continue

            thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][j], nodePositions[1][j])))
            if thisPairNorm < R:
                # add line to plot for part 4
                plt.plot([nodePositions[0][i], nodePositions[0][j]],[nodePositions[1][i], nodePositions[1][j]], color = "blue", linewidth = "0.5")
    plt.scatter(nodePositions[0], nodePositions[1], marker = ">", color = "magenta")
    plt.gcf().gca().set_aspect("equal")
    plt.savefig(fileName)

def plotNodes(nodePositions):

    # plt.clf()
    for i in range(0, NUM_NODES):
        for j in range(0, NUM_NODES):

            # nodes should not be used with themselves
            if i == j:
                continue

            thisPairNorm = euclidean_norm(tuple_diff((nodePositions[0][i], nodePositions[1][i]), (nodePositions[0][j], nodePositions[1][j])))
            if thisPairNorm < R:
                # add line to plot for part 4
                plt.plot([nodePositions[0][i], nodePositions[0][j]],[nodePositions[1][i], nodePositions[1][j]], color = "blue", linewidth = "0.5")
    plt.scatter(nodePositions[0], nodePositions[1], marker = ">", color = "magenta")
    plt.gcf().gca().set_aspect("equal")

def plotAndSaveNodeTrajectories(allPositions, fileName):
    plt.clf()
    # shape[0] is x, y
    # shape[1] is each node
    # shape[2] is each time step
    for i in range(0, allPositions.shape[1]):
        thisNodeX = allPositions[0, i, :]
        thisNodeY = allPositions[1, i, :]
        plt.plot(thisNodeX, thisNodeY, color = "black")
        plt.title("Node Trajectories")
    plt.scatter(allPositions[0, :, allPositions.shape[2] - 1], allPositions[1, :, allPositions.shape[2] - 1], marker=">", color = "magenta")
    plt.gcf().gca().set_aspect("equal")
    plt.savefig(fileName)

def plotAndSaveNodeVelocities(allVelocities, fileName, fileName2):
    plt.clf()
    for i in range(0, allVelocities.shape[1]):
        thisNodeX = allVelocities[0, i, 1:]
        thisNodeY = allVelocities[1, i, 1:]
        mag = np.sqrt(np.add(np.power(thisNodeX, 2), np.power(thisNodeY, 2)))
        plt.plot(mag)
    plt.title("Node Velocities")
    plt.savefig(fileName)

    plt.clf()
    x_vels = allVelocities[0, :, :]
    x_sums = np.sum(x_vels, axis = 0)
    x_sums = np.divide(x_sums, allVelocities.shape[1])
    y_vels = allVelocities[1, :, :]
    y_sums = np.sum(y_vels, axis = 0)
    y_sums = np.divide(y_sums, allVelocities.shape[1])
    avg_mag = np.sqrt(np.power(x_sums, 2) + np.power(y_sums, 2))
    plt.plot(avg_mag)
    plt.title("Average Velocity of All Nodes")
    plt.savefig(fileName2)

def plotAndSaveConnectivity(allPositions, fileName):
    plt.clf()
    # loop for every time point
    connectivity = []
    for i in range(0, allPositions.shape[2]):
        emptyMatrix = np.zeros((allPositions.shape[1], allPositions.shape[1]))
        # loop through every node to make the matrix
        for j in range(0, allPositions.shape[1]):
            for k in range(0, allPositions.shape[1]):
                if j == k:
                    continue
                thisPairNorm = euclidean_norm(tuple_diff((allPositions[0][j][i], allPositions[1][j][i]), (allPositions[0][k][i], allPositions[1][k][i])))
                if thisPairNorm < R:
                    emptyMatrix[j][k] = 1

        rank = np.linalg.matrix_rank(emptyMatrix)
        connectivity.append((1 / NUM_NODES) * rank)
    plt.plot(connectivity)
    plt.title("Connectivity")
    plt.savefig(fileName)

def plotCenterOfMassAndTarget(targetX, targetY, nodePositions, filename):
    plt.clf()
    x_vels = nodePositions[0, :, :]
    x_sums = np.sum(x_vels, axis = 0)
    x_sums = np.divide(x_sums, nodePositions.shape[1])
    y_vels = nodePositions[1, :, :]
    y_sums = np.sum(y_vels, axis = 0)
    y_sums = np.divide(y_sums, nodePositions.shape[1])
    plt.plot(x_sums, y_sums, color = "black", label = "Center of Mass")
    plt.plot(targetX, targetY, color = "red", label = "Gamma Agent")
    plt.legend(loc = "lower left")
    plt.title("Center of Mass vs. Gamma Agent")
    plt.savefig(filename)


if __name__ == "__main__":
    main()
