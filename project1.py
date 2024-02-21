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
Q_MT = (25, 25)
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

def main():

    # initialize nodes
    np.random.seed(17)
    # nodePositions = np.random.randint(low = 0, high = 50, size = (DIMENSIONS, NUM_NODES))
    # nodeVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    # nodeAccelerations = np.zeros((DIMENSIONS, NUM_NODES))
    # initalize holding of everything
    # allPositions = nodePositions.copy()
    # allNodeVelocities = nodeVelocities.copy()
    # allNodeAccelerations = nodeAccelerations.copy()

    camera = Camera(plt.figure(dpi = 300))


    # for i in range(0, 1000):
    #     nodePositions, nodeVelocities, nodeAccelerations = updateAlgo1(nodePositions, nodeVelocities)
    #     plotNodes(nodePositions)
    #     camera.snap()
    #     print("FINISHED ITERATION: ", i)

    # anim = camera.animate(blit=True)
    # anim.save('scatter.mp4')

    # plotNodesAndSave(nodePositions, "start2.png")
    # for i in range(0, 500):
    #     nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2(nodePositions, nodeVelocities)
    #     plotNodes(nodePositions)
    #     camera.snap()
    #     print("FINISHED ITERATION: ", i)
    # plotNodesAndSave(nodePositions, "end2.png")
        
    # anim = camera.animate(blit=True)
    # anim.save("part2-close-more.mp4")

    ##### ALGO 2 - CASE 2 - SINE WAVE #####

    # nodePositions = np.random.randint(low = 0, high = 150, size = (DIMENSIONS, NUM_NODES))
    # nodeVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    # nodeAccelerations = np.zeros((DIMENSIONS, NUM_NODES))
    # allPositions = nodePositions.copy()
    # allNodeVelocities = nodeVelocities.copy()
    # allNodeAccelerations = nodeAccelerations.copy()


    # gammaX = np.arange(500)
    # gammaY = (np.sin(2 * np.pi * 2 * gammaX / 500) * 75) + 75

    # for i in range(0, 500):
    #     if i == 0:
    #         nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), (0, 0))
    #     else:
    #         nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), ((gammaX[i] - gammaX[i - 1])/DELTA_T, (gammaY[i] - gammaY[i - 1])/DELTA_T))
    #     plotNodes(nodePositions)
    #     plt.plot(gammaX, gammaY)
    #     plt.scatter(gammaX[i], gammaY[i],color = "red")
    #     camera.snap()
    #     print("FINISHED ITERATION: ", i)
    # anim = camera.animate(blit = True)
    # anim.save("part3.mp4")

    ##### ALGO 2 - CASE 3 - CIRCLE #####

    nodePositions = np.random.randint(low = 0, high = 150, size = (DIMENSIONS, NUM_NODES))
    nodeVelocities = np.zeros((DIMENSIONS, NUM_NODES))
    nodeAccelerations = np.zeros((DIMENSIONS, NUM_NODES))
    allPositions = nodePositions.copy()
    allNodeVelocities = nodeVelocities.copy()
    allNodeAccelerations = nodeAccelerations.copy()


    thetaOne = np.linspace(1 * np.pi, 0, 250)
    thetaTwo = np.linspace(2 * np.pi, 1 * np.pi, 250)
    theta = np.append(thetaOne, thetaTwo)
    radius = 150
    gammaX = radius * np.cos(theta)
    gammaX = gammaX + 250
    gammaY = radius * np.sin(theta)
    gammaY = gammaY + 150

    for i in range(0, 500):
        if i == 0:
            nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), (0, 0))
        else:
            nodePositions, nodeVelocities, nodeAccelerations = updateAlgo2DynamicTarget(nodePositions, nodeVelocities, (gammaX[i], gammaY[i]), ((gammaX[i] - gammaX[i - 1])/DELTA_T, (gammaY[i] - gammaY[i - 1])/DELTA_T))
        plotNodes(nodePositions)
        plt.plot(gammaX, gammaY)
        plt.scatter(gammaX[i], gammaY[i],color = "red")
        camera.snap()
        print("FINISHED ITERATION: ", i)
    anim = camera.animate(blit = True)
    anim.save("part4.mp4")







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
        if i == 98:
            print("ACCEL", i, newAccel)
            print("VELOCITY X", newVelocities[0][i])
            print("VELCOITY Y", newVelocities[1][i])
            print("POS: ", newPositions[0][i], newPositions[1][i], '\n')
    
    posSumX = 0
    posSumY = 0
    for i in range(0, NUM_NODES):
        posSumX += newPositions[0][i]
        posSumY += newPositions[1][i]
    posSumX /= NUM_NODES
    posSumY /= NUM_NODES
    print("CENTER OF MASS: ", posSumX, posSumY)

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
        if i == 98:
            print("ACCEL", i, newAccel)
            print("VELOCITY X", newVelocities[0][i])
            print("VELCOITY Y", newVelocities[1][i])
            print("POS: ", newPositions[0][i], newPositions[1][i], '\n')
    
    posSumX = 0
    posSumY = 0
    for i in range(0, NUM_NODES):
        posSumX += newPositions[0][i]
        posSumY += newPositions[1][i]
    posSumX /= NUM_NODES
    posSumY /= NUM_NODES
    print("CENTER OF MASS: ", posSumX, posSumY)

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

if __name__ == "__main__":
    main()
