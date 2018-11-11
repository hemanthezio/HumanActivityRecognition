# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 07:21:19 2018

@author: Hemanth kumar
"""

import matplotlib.pyplot as plt
from progressbar import ProgressBar
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

class DeepNetwork:
    
    ''' Deep Network by Hemanth kumar '''
    
    def __init__(self,X_train,y_train):
        
        '''     Initialize the network with training  data    
            X_train     --> InDependent variable
            y_train     --> Dependent variable
        
        '''
        self.X_train=X_train
        self.y_train=y_train
        
        print("\nNetwork Initialized..........\n")
        print("Number of Training samples :      ",len(X_train[0]))
        print("Number of Features :             ",len(X_train))
        print("Available Activations :\n1.relu\n2.tanh\n3.sigmoid\n")
    
    
    # Function to perforn K-FOLD CROSS VALIDATION
    def k_fold_crossval(self,af,layer_dim,epoch,batch_size,alpha,k):
        '''
            af          --> list of activation function    [AF in hidden layer,AF of output unit]
            layer_dim   --> Dimension of the network , list of number of nodes at each layer [n1,n2,n3]
            epochs      --> Max Epochs / Iterations
            batch_size  --> Batch size
            alpha       --> Learning rate
            K           --> Fold size
            
        '''
        print("\n\nK-Fold Cross Validation initiated...........\n")
        print("Alpha :               ",alpha)
        print("Network Dimension :   ",layer_dim)
        print("Activations :         ",af)
        print("Batch Size :          ",batch_size)
        print("Fold size :           ",k,"\n\n")
        
        
        x=self.X_train.T
        y=self.y_train.T
        scores=[]
    
        kf = KFold(n_splits=k)
        kf.get_n_splits(x)
        i=1
        for train_index, test_index in kf.split(x):
            X__train, X__test = x[train_index], x[test_index]
            y__train, y__test = y[train_index], y[test_index]

        
            para,J_log=self.network(X__train.T,y__train.T,af,layer_dim,epoch,batch_size,alpha,plot=False,print_info=False)
            y_pred=self.predict(X__test.T,len(layer_dim),para,af)
            accuracy = accuracy_score(y__test,y_pred.T)
            f1=f1_score(y__test,y_pred.T)
            print('\nFold '+str(i)+' Accuracy Score:',accuracy*100,'%')
            print('\nFold '+str(i)+' F1 Score:',f1,'\n')
            i+=1
            scores.append(accuracy)
        
        return scores
    
    # Function to define sigmoid activation function
    def sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))
        return s

    # Function to define tanh activation function
    def tanh(self,z):
        return np.tanh(z)
   
    # Function to define relu activation function
    def relu(self,z):
        z=np.where(z<0,0,z)
        return z
    
    # Function to initialize deep network
    def init_deep(self,layer_size):
        '''
        layer_size  --> list of number of nodes at each layer
        
        '''
    
        np.random.seed(3)
        parameters = {}
        n = len(layer_size)

        for l in range(1, n):
            parameters['W' + str(l)] = np.random.randn(layer_size[l], layer_size[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_size[l], 1))
        
        return parameters
    
    # Function to perform forward propogation
    def forward_prop(self,W,X,b,AF):
        '''
            W   --> Weight matrix of np
            X   --> Independent Variable
            b   --> Bias
            AF  --> Activation Function
        '''
        Z=np.dot(W,X)+b
        if(AF=='tanh'):
            A=self.tanh(Z)
        elif AF=='sigmoid':
            A=self.sigmoid(Z)
        elif AF=='relu':
            A=self.relu(Z)
        return (Z,A)
    
    # Function to define derivatives of activation function
    def dir_AF(self,A,AF):
        '''
            AF  --> Activation Function of whic derivative has to be found
            A   --> np vector to which derivative has to be calculated using Activation function
        '''
        
        if AF=='tanh':
            return 1 - np.power(A, 2)
        
        elif AF=='sigmoid':
            return A*(1-A)
        
        elif AF=='relu':
            A=np.where(A>=0,1,A)
            A=np.where(A<0,0,A)
            return A
    
    # Function to predict 
    def predict(self,X,layer_size,para,af):
        '''
            X           --> InDependent variable used for prediction
            layer_size  --> List of number of nodes at each layer
            para        --> Parameters of the network
            af          --> List of activation function    [AF in hidden layer,AF of output unit]
        '''
        
        Z={}
        A={}
        A['A0']=X
        for l in range(1,layer_size):
            if l==layer_size-1:
                AF=af[1]
            else:
                AF=af[0]
            Z['Z'+str(l)],A['A'+str(l)]=self.forward_prop(para['W'+str(l)],A['A'+str(l-1)],para['b'+str(l)],AF)
        y_pred=np.round(A['A'+str(layer_size-1)])
        return y_pred

    #Function to plot error curve for each batch
    def plot_graph(J_log):
        '''
            J_log   --> List of errors per batch
        '''
        
        plt.title('Cost V/S Batch no#')
        plt.xlabel('Batch no#')
        plt.ylabel('Cost')
        plt.plot(range(1,len(J_log)+1,1), J_log)
        plt.show()
    
    
    # Function that defines DNN
    def network(self,x,y,af,l_dim,epoch,batch_size,alpha=0.2,plot=True,print_info=True):
        '''
            Parameters:
            x           --> Train data InDependent variable
            y           --> Train data Dependent variable
            af          --> List of activation function    [AF in hidden layer,AF of output unit]
                           Available af:
                                   1.relu
                                   2.tanh
                                   3.sigmoid
            l_dim       --> Dimension of the network , list of number of nodes at each layer [n1,n2,n3]
            epoch       --> Max Epochs / Iterations
            batch_size  --> Size of batch
            alpha       --> Learning rate
            plot        --> if True plots graph of error of each batch
            
            return value:
            1.parameters of the network i.e weights and bias at each layer as a dictionary
            2.cost or error at each Epoch or iteration
            
        '''
        if(print_info):
            print("\n\nTraining the Deep Network ......\nInitiated................\n")
            print("Network Dimension :   ",l_dim)
            print("Activations :         ",af)
            print("Alpha :               ",alpha)
            print("Batch Size :          ",batch_size)
            print("Epochs :              ",epoch,"\n")
        
        
        pbar = ProgressBar()
        layer_dim=l_dim.copy()
        layer_size=len(layer_dim)
        
        ''' Initialize Deep Network '''
        para=self.init_deep(layer_dim)
        
        m=len(x[0])
        J_log=[]
        J_log_outer=[]
        
        #Calulate inital error of network
        a={}
        z={}
        a['A0']=x
        for l in range(1,layer_size):
            if l==layer_size-1:
                AF=af[1]
            else:
                AF=af[0]
            z['Z'+str(l)],a['A'+str(l)]=self.forward_prop(para['W'+str(l)],a['A'+str(l-1)],para['b'+str(l)],AF)
        J = (- 1 / m) * np.sum(y * np.log(a['A'+str(layer_size-1)]) + (1 - y) * (np.log(1 - a['A'+str(layer_size-1)])))   
        J_log_outer.append(J)
    
        ''' Main Loop '''
        for i in pbar(range(epoch)):
            J_log=[]
            '''---------Forward propogation---------'''
        
            for b in list(range(0,m,batch_size)):
                # BATCH TRAIN BEGIN
                Z={}
                A={}
                dZ={}
                dW={}
                dA={}
                dB={}
                if(b+batch_size>m):
                    ss=(m-b+batch_size)%batch_size
                    select=list(range(b,b+ss))
                else:
                    select=list(range(b,b+batch_size))
                
                #Initial Error of Batch
                A['A0']=x[:,select]
                for l in range(1,layer_size):
                    if l==layer_size-1:
                        AF=af[1]
                    else:
                        AF=af[0]
                    Z['Z'+str(l)],A['A'+str(l)]=self.forward_prop(para['W'+str(l)],A['A'+str(l-1)],para['b'+str(l)],AF)   
                
                if(plot==True):
                    J = (- 1 / batch_size) * np.sum(y[:,select] * np.log(A['A'+str(layer_size-1)]) + (1 - y[:,select]) * (np.log(1 - A['A'+str(layer_size-1)])))
                    J_log.append(J)
                
                
                '''---------Back propogation-------'''
                
                # Initialize Back prop
                dA['A'+str(layer_size-1)] = - (np.divide(y[:,select], A['A'+str(layer_size-1)]) - np.divide(1 - y[:,select], 1 - A['A'+str(layer_size-1)]))
                for l in reversed(range(layer_size)):
                    if(l==0):
                        break
                    if l==layer_size-1:
                        AF=af[1]
                    else:
                        AF=af[0]
                    # Calculate Derivatives
                    dZ['Z'+str(l)]=dA['A'+str(l)] * self.dir_AF(A['A'+str(l)],AF)
                    dW['W'+str(l)]=(1/batch_size)*np.dot(dZ['Z'+str(l)],A['A'+str(l-1)].T)
                    dB['B'+str(l)]=(1/batch_size)*(np.sum(dZ['Z'+str(l)],axis=1,keepdims=True))
                    dA['A'+str(l-1)]=np.dot(para['W'+str(l)].T,dZ['Z'+str(l)])
                para_tmp={}
                
                # Update weights and bias
                for l in range(1, layer_size):
                    para_tmp['W' + str(l)]= para['W' + str(l)] - alpha*dW['W' + str(l)]
                    para_tmp['b' + str(l)]= para['b' + str(l)] -  alpha*dB['B' + str(l)]
                para=para_tmp
        
            if(plot==True):
                self.plot_graph(J_log)
            
            # Calculate cost/error per epoch or iteration 
            a={}
            z={}
            a['A0']=x
            for l in range(1,layer_size):
                if l==layer_size-1:
                    AF=af[1]
                else:
                    AF=af[0]
                z['Z'+str(l)],a['A'+str(l)]=self.forward_prop(para['W'+str(l)],a['A'+str(l-1)],para['b'+str(l)],AF)
            J = (- 1 / m) * np.sum(y * np.log(a['A'+str(layer_size-1)]) + (1 - y) * (np.log(1 - a['A'+str(layer_size-1)])))
            J_log_outer.append(J)
        
        return para,J_log_outer

