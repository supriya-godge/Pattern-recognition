'''
Name: Supriya Godge
Assignment 1: To make a linear and KNN classifier.
'''

from pylab import *
import math
import numpy as np

class data:
    __slots__ = 'x_val','y_val','label','dist'

    def __init__(self,x,y,l):
        self.x_val = x
        self.y_val = y
        self.label = l
        self.dist  = math.inf

    def __str__(self):
        return self.x_val+" "+self.y_val+" "+self.label

    def __repr__(self):
        return str(self.x_val) + " " + str(self.y_val) + " " + str(self.label)+"\n"

def start():
    '''
    This function reads the input and calls the classifier functions
    :return: None
    '''
    dataBase = load("data.npy")
    dataList=[]

    size = dataBase.__len__()
    #print(dataBase)
    #print("_________________________________")
    label = np.zeros((size,1),dtype=np.float)
    X = np.zeros((size, 3), dtype=np.float)
    index=0
    for iter in dataBase:
        print(iter)
        dataList.append(data(iter[0],iter[1],iter[2]))
        X[index][0]=1
        X[index][1]=iter[0]
        X[index][2]=iter[1]
        label[index] = iter[2]
        index+=1



    X1= np.matrix(X)
    label1 = np.matrix(label)

    rpl =True
    while(rpl):
        print("=================Classification=========================")
        print("1. KNN 1")
        print("2. KNN 15")
        print("3. Linear")
        print("4. Quit")
        ans = int(input("Enter your choice (1,2 or 3)"))
        if ans==2 or ans==3 or ans==1:
            if ans==1:
                KNN(1, dataList, X)
            if ans==2:
                KNN(15, dataList, X)
            if ans==3:
                LinearModel(X1, label1, dataList)
        else:
            rpl = False







def findLabelLinear(current, beta):
    '''
    This function finds the correct class label given the point
    coordinates and beta values
    :param current: Point X,Y coordinate
    :param beta: Beta values
    :return: classified Class
    '''
    Y= current.transpose() * beta


    if Y >= 0.5:
        return "co",1
    else:
        return "mo",0

def LinearModel(X,label,dataList):
    '''
    This is the linear classification fun
    :param X: X and Y coordinator numpy array
    :param label: Target variable array
    :param dataList: List of objects
    :return: None
    '''
    X_T = X.transpose()
    multi = X_T * X

    X_inv = inv(multi)
    beta_temp = X_inv * X_T
    beta = beta_temp * label
    #print(beta)

    minX = min(X[:,1])
    minY = min(X[:,2])
    maxX = max(X[:,1])
    maxY = max(X[:,2])


    X1 = np.arange(-3, 5, 0.08)
    Z = np.zeros((len(X1)*len(X1)),dtype=np.int)

    index=0
    xCordinate =0
    while xCordinate < len(X1):
        yCordinate = 0
        while yCordinate < len(X1):
            #print(xCordinate,' ',yCordinate)
            current = np.array([1,X1[xCordinate], X1[yCordinate]])
            label1 = findLabelLinear(current, beta)
            Z[index] = label1[1]
            if label1[1]==0:
                plt.scatter(X1[xCordinate], X1[yCordinate], color="RED",alpha=0.1,edgecolors=None)
            else:
                plt.scatter(X1[xCordinate], X1[yCordinate], color="BLUE",alpha=0.1,edgecolors=None)
            yCordinate += 1
            index+=1
        xCordinate += 1

    iter=0


    X1 = np.squeeze(np.asarray(X1))
    label= np.squeeze(np.asarray(label))
    #print(label)
    for item in dataList:
        #print(iter)
        if label[iter] == 0:
            plt.scatter(item.x_val, item.y_val, s=80, facecolors='none', edgecolors='r')
        else:
            plt.scatter(item.x_val, item.y_val, s=80, facecolors='none', edgecolors='b')
        iter+=1

    h,w = np.meshgrid(X1,X1)
    Z=Z.reshape(h.shape)

    plt.plot([], 'ro', label='class1')
    plt.plot([], 'bo', label='class2')
    plt.xlabel('X-Axis')
    plt.contour(w, h, Z, 1, colors='k')
    plt.ylabel('Y-Axis')
    plt.title(" Linear Classification")
    plt.legend()
    plt.show()
    confusionMatrix(dataList, "linear", 0, beta)


def confusionMatrix(dataList,type,n,beta=None):
    '''
    This function creates the confusion matrix
    :param dataList: The list of the data objects
    :param type: Type of classification (KNN/Linear)
    :param n: number of neighbors
    :param beta: Beta values
    :return: None
    '''
    correct = 0
    total = 0
    classActual0 = 0
    classPredict0 = 0
    classActual1=0
    ClassA0P1 = 0
    ClassA1P0 = 0
    count=0
    dataList1=dataList[:]
    for val in dataList1:
        if type =="knn":
            la = findLabel(val.x_val, val.y_val, n, dataList)
        if type=="linear":
            current = np.array([1, val.x_val, val.y_val])
            la = findLabelLinear(current,beta)
        if val.label == 0:
            classActual0 += 1
            if la[1] == 1:
                ClassA0P1 += 1
        if la[1] == 0:
            classPredict0 += 0
        if val.label == 1:
            classActual1+=1
            if la[1] == 0:
                ClassA1P0 += 1

        if la[1] == val.label:
            correct += 1
        total += 1

    print(classActual0, " ", classActual1)
    print("                        Predicted Value                  ")
    print("______________________________________________________")
    print("|                     |  Class 0      | Class1         |")
    print("|_____________________|_______________|________________|")
    print("|Actual |  Class 0    |     ",end="")
    print(classActual0 - ClassA0P1,end="")
    print("        |      ",end="")
    print(ClassA0P1,end="")
    print("        |")
    print("| Value |_____________|_______________|________________|")
    print("|       |  Class 1    |     ",end="")
    print(ClassA1P0,end="")
    print("        |       ",end="")
    print(classActual1-ClassA1P0,end="")
    print("       |")
    print("|_______|_____________|_______________|________________|")

    print("Accuracy:",correct/total*100)




def KNN(n,dataList,X):
    minX = (min(item.x_val for item in dataList))
    minY = (min(item.y_val for item in dataList))
    maxX = (max(item.x_val for item in dataList))
    maxY = (max(item.y_val for item in dataList))
    X = np.arange(-3, 5, 0.08)
    Z = np.zeros((len(X)*len(X)),dtype=np.int)


    confusionMatrix(dataList,"knn",n)

    final=[]
    index=0
    xCordinate =0
    #while xCordinate < int(maxX+1):
    while xCordinate < len(X):
        yCordinate =0
        #while yCordinate < int(maxY+1):
        while yCordinate < len(X):
            #print(xCordinate,' ',yCordinate)
            label = findLabel(X[xCordinate],X[yCordinate], n, dataList)
            Z[index]=label[1]
            #plt.plot(X[xCordinate], X[yCordinate], label[0])
            if label[1]==0:
                plt.scatter(X[xCordinate], X[yCordinate], color="RED",alpha=0.1,edgecolors=None)
            else:
                plt.scatter(X[xCordinate], X[yCordinate], color="BLUE",alpha=0.1,edgecolors=None)
            yCordinate += 1
            index+=1
        xCordinate += 1
    iter=0
    for item in dataList:
        #print(iter)
        iter+=1
        if item.label == 0:
            plt.scatter(item.x_val, item.y_val, s=80, facecolors='none', edgecolors='r')
        else:
            plt.scatter(item.x_val, item.y_val, s=80, facecolors='none', edgecolors='b')

    plt.plot([], 'r', label='class1')
    plt.plot([], 'b', label='class2')
    xCordinate=0
    yCordinate=0

    h,w = np.meshgrid(X,X)
    Z=Z.reshape(h.shape)

    plt.contour(w,h,Z,1,colors='k')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.title(str(n)+" NN")
    plt.legend()
    plt.show()







def findLabel(x,y,n,dataList):
    for iter in dataList:
    #for iter in dataList[:dataList.__len__()-15]:
        #min = -math.inf
        iter.dist = math.sqrt(math.pow(x-iter.x_val,2)+math.pow(y-iter.y_val,2))

    dataList.sort(key=lambda a:a.dist)

    L0=0
    L1=0
    for iter in range(n):
        if dataList[iter].label == 0:
            L0=L0+1
        else:
            L1=L1+1

    val0 = L0/(L1+L0)
    val1 = L1 / (L1 + L0)

    if val0>0.50:
        return "mo",0
    else:
        return "co",1
    #else:
    #    return "ko"













if __name__ == '__main__':
    start()