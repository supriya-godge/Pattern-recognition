'''
Name: Supriya Godge
Assignment 2: Create a second program, a2cost.py. This program will use your (approximated)
Bayesian classifier from Question 1, but with different cost functions. Using the provided
nuts_bolts.csv file from “Classification, Parameter Estimation and State Estimation,”
have the program create the plots shown in Figure 2.5a) and c) (i.e. using the cost function
on page 20, and using the the 0-1 loss cost function), and have your program save these
to disk as files.
'''


import numpy as np
from numpy.linalg import inv
import math
from numpy.linalg import det
import matplotlib.pyplot as plt


def readFile(fileName):
    '''
    this function reads the input file and creates the numpy array
    to store ots values.
    :param fileName: Name of the file
    :return: The numpy array with file values
    '''
    data_list = []
    bolt=nut=ring=scrap=0   #Specific to the problem at hand
    with open(fileName) as fileDiscriptor:
        for line in fileDiscriptor:
            line = line.strip();
            line = line.split(",")
            inner_list =[]
            for iter in line:
                inner_list.append(float(iter))
            data_list.append(inner_list)

    data = np.array(data_list)
    #print(data_list)
    bolt = np.sum(data[:, 2] == 1)
    nut = np.sum(data[:,2] == 2)
    ring = np.sum(data[:, 2] == 3)
    scrap = np.sum(data[:, 2] == 4)
    ring=20*ring
    scrap *= 2
    return data,bolt,nut,ring,scrap




def  gaussianTwo(dataPair,C_k,mean_val,first_term):
    '''
    Calculates the conditional probability density function
    :param feature: The mesaurement vector
    :param className: Uniqie class identifier
    :return: conditional probability density
    '''
    ans=[]
    #print("ft:",first_term)
    featuresubMean = dataPair - mean_val
    featuresubMean = np.asmatrix(featuresubMean)
    expo=((-1*featuresubMean)*inv(C_k)*featuresubMean.transpose())/2
    exponent=np.power(math.e,expo)
    val = first_term * exponent
    ans.append(val.item(0))
    return np.asarray(ans)


def GaussianMatrix(x,z,sigma,axis=None):
    return np.exp((-(np.linalg.norm(x-z, axis=axis)**2))/(2*sigma**2))

x=np.arange(12).reshape(3,4)


def gaussian_helperFunction(feature,className):
    feature = feature[feature[:, 2] == className]
    feature = feature[:, :2]
    total_len = len(feature[:, 0])
    mean0 = np.sum(feature[:, 0]) / total_len
    mean1 = np.sum(feature[:, 1]) / total_len
    mean_val = np.array([mean0, mean1])
    C_k = covarience(feature)
    first_term = 1 / (math.sqrt(((2 * math.pi) ** 4) * det(C_k)))
    return C_k,mean_val,first_term


def covarience(feature):
    total_len = len(feature[:, 0])
    feature = np.asmatrix(feature)
    ones = np.matrix(np.ones((total_len, total_len)))
    substepone = feature-(ones * feature)/total_len
    substeptwo = substepone.transpose()* substepone / (total_len-1)
    return substeptwo


def classifyExistingData(class_all,data,cost_matrix,priorProbability):
    classifiedPoints=[]
    color =["r","g","y","b"]
    mark = ["+","*","o","x"]
    class_confusion=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    values0=[]
    values1=[]
    for item in data:
        rows = item[:2]
        merge = []
        for class_val in range(0, 4):
            val1 = gaussianTwo(rows, class_all[class_val][0], class_all[class_val][1], class_all[class_val][2]) \
                   * priorProbability[class_val]
            merge.append(val1)

        merge_mat = np.asmatrix(merge)
        final = merge_mat.transpose() * cost_matrix.transpose()
        class_final = np.argmin(final) + 1
        classifiedPoints.append([rows, class_final])
        actual_class= item[2]
        class_final  = class_final.astype(int)
        actual_class = actual_class.astype(int)
        class_confusion[int(class_final-1)][actual_class-1]+=1
        values0.append(rows[0])
        values1.append(rows[1])
        #plt.xlim(-2,1)
        #plt.ylim(-2, 1)

        plt.scatter(rows[0], rows[1], color="black", marker=mark[class_final-1],edgecolors=None)
    #plt.scatter(values0, values1, color="black", marker="x", edgecolors=None)

    #print(class_confusion)
    return class_confusion

def  createcostMatrix(choice="non 0-1"):
    '''
    This function creates the cost matrix
    :param choice: There are to choices 1. Normal cost mtrix as per reference book
     choice 2: 0-1 cost matrix
    :return: Cost Matrix
    '''
    cost_matrix = np.ones((4, 4))
    if choice == "non 0-1":
        cost_matrix[0][0] = -0.20
        cost_matrix[1][1] = -0.15
        cost_matrix[2][2] = -0.05
        cost_matrix[3][3] = cost_matrix[3][0] = cost_matrix[3][1] = 0.03
        cost_matrix[0][1] = cost_matrix[0][2] = cost_matrix[0][3] = cost_matrix[1][0] = cost_matrix[1][2] \
            = cost_matrix[1][3] = cost_matrix[2][0] = cost_matrix[2][1] = cost_matrix[2][3] = 0.07
    if choice == "0-1":
        cost_matrix[0][0] = cost_matrix[2][2] = cost_matrix[1][1] = cost_matrix[3][3]=0
        cost_matrix[3][0] = cost_matrix[3][1] = 1
        cost_matrix[0][1] = cost_matrix[0][2] = cost_matrix[0][3] = cost_matrix[1][0] = cost_matrix[1][2] \
            = cost_matrix[1][3] = cost_matrix[2][0] = cost_matrix[2][1] = cost_matrix[2][3] = 1

    return cost_matrix


def createTheBasicMeanFunctions(data):
    class1 = gaussian_helperFunction(data, 1)
    class2 = gaussian_helperFunction(data, 2)
    class3 = gaussian_helperFunction(data, 3)
    class4 = gaussian_helperFunction(data, 4)
    class_all = [class1, class2, class3, class4]
    return class_all


def start(title_img,costM=None):
    '''
    This function implements the main Bayesian algorithm
    :return:None
    '''
    print()
    print("--------------------------------------------------------------")
    print("\t\t\t\t\t  ",title_img.strip(".png"))
    print("--------------------------------------------------------------")
    data_features, bolt, nut, ring, scrap = readFile("nuts_bolts.csv")
    data = data_features[:]
    total = bolt+nut+ring+scrap
    priorProbability = np.array([bolt/total,nut/total,ring/total, scrap/total])

    #print("Prior Probabilities", priorProbability)
    cost_matrix = createcostMatrix(costM)
    #print("cost Matrix",cost_matrix)
    points = np.arange(-0.2, 1, 0.01)
    class_all= createTheBasicMeanFunctions(data)
    iter=0
    classifiedPoints=[]

    Z=[]

    classified_contor=[]
    for iter in range(len(points)):
        prev =None
        for jiter in range(len(points)):
            rows = [points[iter],points[jiter]]
            merge=[]
            for class_val in range(0,4):
                val1 =  gaussianTwo(rows,class_all[class_val][0], class_all[class_val][1],class_all[class_val][2])
                val1 = val1 * priorProbability[class_val]
                merge.append(val1)
            merge_mat = np.asmatrix(merge)
            final = merge_mat.transpose() * cost_matrix.transpose()
            class_final = np.argmin(final) + 1
            if prev != None and prev != class_final:
                classified_contor.append([rows[0],rows[1]])
            prev = class_final
            classifiedPoints.append([rows, class_final])
            Z.append(class_final)


    for iter in classifiedPoints:
        class_final = iter[1]
        rows = iter[0]
        if class_final==1:
            plt.scatter(rows[0], rows[1], color="RED",alpha=0.1,edgecolors=None)
        if class_final==2:
            plt.scatter(rows[0], rows[1], color="GREEN",alpha=0.1,edgecolors=None)
        if class_final==3:
            plt.scatter(rows[0], rows[1], color="Y",alpha=0.1,edgecolors=None)
        else:
            plt.scatter(rows[0], rows[1], color="BLUE",alpha=0.1,edgecolors=None)

    for iter in classified_contor:
        plt.scatter(iter[0], iter[1],s=10, color="black", alpha=1, edgecolors=None)

    confusion_matrix_data = classifyExistingData(class_all,data,cost_matrix,priorProbability);
    confusion_matrix_helper(confusion_matrix_data)
    plt.scatter([],[],marker="+", c='r', label='Bolt')
    plt.scatter([],[],marker="x", c='b', label='Srap')
    plt.scatter([],[],marker="o", c='y', label='Ring')
    plt.scatter([],[],marker="*", c='g', label='Nut')

    plt.xlabel('measure of six-fold rotational symmetry')

    plt.legend()
    plt.ylabel('measure of eccentricity')
    #print(merge)
    plt.savefig(title_img)
    #plt.show()
    plt.cla()

def confusion_matrix_helper(confusion_matrix_data):
        title = ["bolt", "nut ", "ring", "scrap"]
        confusionMatrix(title,confusion_matrix_data)


def confusionMatrix(trueTitle,data):
    '''
    This function creates and printes the confusion matrix
    :param trueTitle: Class names
    :param data: infomation of data beging classified into different classes
    :return: None
    '''
    print("\t\t\t\t\t\t\t\t    Actual Values   ")
    a=["Expected"," value  "]
    iter=0
    jiter=0
    print("\t\t\t\t\t     ",end=" ")
    #Printing the expected value classes
    for i in trueTitle:
        print(i, end="\t ")
        data[iter]=[i]+data[iter] # appending the expected class name
        if jiter< len(a):
            data[iter] = [a[jiter]]+data[iter]
        else:
            data[iter] = ["        "] + data[iter]
        iter+=1
        jiter+=1
    #Printing the confusion matrix
    for row in data:
        print()
        for val in row:
            print(val,end="\t\t")





if __name__ == '__main__':
    start("Bayesian_with_0-1Cost.png","0-1")
    start("Bayesian_with_NormalCost.png", "non 0-1")



