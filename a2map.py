'''
Name: Supriya Godge
Assignment 2: Implement a maximum a posteriori (MAP) classifier, using Gaussian distributions
to estimate class-conditional densities. Create a program a2map.py that reads in the
data.npy file from Assignment 1, and creates a plot showing 1) the training data, 2) the
decision boundary separating class 0 and 1, and 3) the classification regions
'''


import numpy as np
from numpy.linalg import inv
import math
from numpy.linalg import det
import matplotlib.pyplot as plt
from pylab import *



def readFile(fileName):
    '''
    this function reads the input file and creates the numpy array
    to store ots values.
    :param fileName: Name of the file
    :return: The numpy array with file values
    '''
    data_list = []
    bolt=nut=ring=scrap=0   #Specific to the problem at hand
    fileDiscriptor = load(fileName)
    for line in fileDiscriptor:
            #print(line)
            #line = line.strip();
            #line = line.split(",")
            inner_list =[]
            for iter in line:
                inner_list.append(float(iter))
            data_list.append(inner_list)

    data = np.array(data_list)
    #print(data_list)
    bolt = np.sum(data[:, 2] == 0)
    nut = np.sum(data[:,2] == 1)
    #ring = np.sum(data[:, 2] == 3)
    #scrap = np.sum(data[:, 2] == 4)
    return data,bolt,nut





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




def gaussian_helperFunction(feature,className,n):
    feature = feature[feature[:, 2] == className]
    feature = feature[:, :2]
    total_len = len(feature[:, 0])
    mean0 = np.sum(feature[:, 0]) / total_len
    mean1 = np.sum(feature[:, 1]) / total_len
    mean_val = np.array([mean0, mean1])
    C_k = covarience(feature)
    #print("covarience",C_k)
    first_term = 1 / (math.sqrt(((2 * math.pi) ** n) * det(C_k)))
    return C_k,mean_val,first_term


def covarience(feature):
    '''

    :param feature:
    :return:
    '''
    total_len = len(feature[:, 0])
    feature = np.asmatrix(feature)
    ones = np.matrix(np.ones((total_len, total_len)))
    substepone = feature-(ones * feature)/total_len
    substeptwo = substepone.transpose()* substepone / (total_len-1)
    return substeptwo


def classifyExistingData(class_all,data,cost_matrix,priorProbability,n):
    classifiedPoints=[]
    color =["red","blue","y","b"]
    mark = ["+","*","o","x"]
    class_confusion=[[0,0],[0,0,]]#,[0,0,0,0],[0,0,0,0]]
    for item in data:
        rows = item[:2]
        merge = []
        for class_val in range(0, n):
            val1 = gaussianTwo(rows, class_all[class_val][0], class_all[class_val][1], class_all[class_val][2])
                  # * priorProbability[class_val]
            merge.append(val1)

        final=merge_mat = np.asmatrix(merge)
        #final = merge_mat.transpose() * cost_matrix.transpose()
        #print(final)
        class_final = np.argmax(final)
        classifiedPoints.append([rows, class_final])
        actual_class= item[2]
        class_final  = class_final.astype(int)
        actual_class = actual_class.astype(int)
        #print(class_final, " ",actual_class )
        class_confusion[int(class_final)][actual_class]+=1
        #print(actual_class," ",color[actual_class])
        plt.scatter(rows[0], rows[1], s=80,facecolor='none',edgecolors=color[actual_class])
        '''
        if actual_class == 0:
            plt.scatter(item.x_val, item.y_val, s=80, facecolors='none', edgecolors='r')
        else:
            plt.scatter(item.x_val, item.y_val, s=80, facecolors='none', edgecolors='b')
        '''



    #print(class_confusion)
    return class_confusion

def  createcostMatrix(choice="non 0-1"):
    '''
    This function creates the cost matrix
    :param choice: There are to choices 1. Normal cost mtrix as per reference book
     choice 2: 0-1 cost matrix
    :return: Cost Matrix
    '''
    cost_matrix = np.ones((2, 2))
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
    if choice =="1-1":
        cost_matrix[0][0]= cost_matrix[1][1]=1
        cost_matrix[0][1] =   cost_matrix[1][0] =1

    return cost_matrix


def createTheBasicMeanFunctions(data,n):
    class1 = gaussian_helperFunction(data, 0,n)
    class2 = gaussian_helperFunction(data, 1,n)
    #class3 = gaussian_helperFunction(data, 3,n)
    #class4 = gaussian_helperFunction(data, 4,n)
    class_all = [class1, class2]
    return class_all


def start():
    '''
    This function implements the main Bayesian algorithm
    :return:None
    '''
    n=2
    data_features, bolt, nut= readFile("data.npy")
    data = data_features[:]
    total = bolt+nut
    priorProbability = np.array([bolt/total,nut/total])
    #print("Prior Probabilities", priorProbability)
    cost_matrix = createcostMatrix("1-1")
    #print("cost Matrix",cost_matrix)
    points = np.arange(-3, 5, 0.08)
    class_all= createTheBasicMeanFunctions(data,n)
    iter=0
    classifiedPoints=[]

    Z=[]
    classified_contor=[]
    prev=None
    for iter in range(len(points)):
        prev=None
        for jiter in range(len(points)):
            rows = [points[iter],points[jiter]]
            merge=[]
            for class_val in range(0,n):
                val1 =  gaussianTwo(rows,class_all[class_val][0], class_all[class_val][1],class_all[class_val][2])
                val1 = val1 * priorProbability[class_val]
                merge.append(val1)
            final=merge_mat = np.asmatrix(merge)
            #final = merge_mat.transpose() * cost_matrix.transpose()
            class_final = np.argmax(final)
            if prev != None and prev != class_final:
                classified_contor.append([rows[0],rows[1]])
            prev = class_final
            classifiedPoints.append([rows, class_final])
            Z.append(class_final)

    for iter in classifiedPoints:
        class_final = iter[1]
        rows = iter[0]
        if class_final==0:
            plt.scatter(rows[0], rows[1], color="RED",alpha=0.1,edgecolors=None)
        if class_final==1:
            plt.scatter(rows[0], rows[1], color="BLUE",alpha=0.1,edgecolors=None)


    confusion_matrix_data = classifyExistingData(class_all,data,cost_matrix,priorProbability,n);
    confusion_matrix_helper(confusion_matrix_data,total)
    h,w = np.meshgrid(points,points)
    Z = np.asarray(Z)
    plt.scatter([],[],marker="o",facecolor='none', edgecolor='r', label='Class1')
    plt.scatter([],[], marker="o", facecolor='none', edgecolor='b', label='Class2')
    Z=Z.reshape(h.shape)
    plt.xlabel('Attribute 1')
    range1 = [ x for x in range(1,n)]
    plt.title("Class1 and class2 classification using MAP")
    #plt.contour(w, h, Z, 1, colors='k')
    for iter in classified_contor:
        plt.scatter(iter[0], iter[1], s=10, color="black", alpha=1, edgecolors=None)
    plt.legend()
    plt.ylabel('Attribute 2')
    #plt.show()
    plt.savefig('a2MAP.png')

def confusion_matrix_helper(confusion_matrix_data,total):
        title = ["class0", "class1"]
        confusionMatrix(title,confusion_matrix_data,total)


def confusionMatrix(trueTitle,data,total):
    '''
    This function creates and printes the confusion matrix
    :param trueTitle: Class names
    :param data: infomation of data beging classified into different classes
    :return: None
    '''
    add=0

    add=data[0][0]+data[1][1]
    print("\t\t\t\t     Actual Values   ")
    a=["Expected"," value  "]
    iter=0
    jiter=0
    print("\t\t\t\t\t    ",end=" ")
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

    print()
    print()
    print("Accuracy: ",add/total*100,"%")







if __name__ == '__main__':
    start()



