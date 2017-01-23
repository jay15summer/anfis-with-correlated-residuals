# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 12:48:52 2016

@author: Jie Ren

An example showing iterative estimation for a spatial model y = ANFIS(x) + GP(S). 

"""

import anfis_co as anfis
from membership import membershipfunction #import membershipfunction 
import numpy
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess
'''
input X , S(spatial location), Y
1. Using anfis with covariance C equals to I to train the model and make predicitions Yp. Input: {Xtrain, Ytrain}
2. input S, Y-Yp to Kriging model and output covariance C. Input: {Strain, Ytrain}
3. Using anfis with covariance C (step 2's output) to train the model and make precitions Yp2. Input: {Xtrain, Ytrain}
4. input S, Y-Yp2 to train a Kriging model
5. use the trained anfis model (step 3) and the trained spatial model (step 4) to make predictions on {Xtest, Stest}, respectively. 
6. Sum up the output in step 5 to get Ytest-predict. Compute the MSE using Ytest_true and Ytest_predict. 
'''
# Loading train and test data
t_train = numpy.loadtxt("traindata.txt", usecols=[0,1,2,3,4,5,6,7])
t_test = numpy.loadtxt("testdata.txt", usecols=[0,1,2,3,4,5,6,7])

## step 1 
Xtrain = t_train[:,0:3]
Ytrain = t_train[:,5]
C = numpy.identity(len(Ytrain))
mf = [[['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':6.5,'sigma':2.}],['gaussmf',{'mean':14.,'sigma':2.}]],
        [['gaussmf',{'mean':1.,'sigma':0.15}],['gaussmf',{'mean':1.5,'sigma':0.15}], ['gaussmf',{'mean':2.,'sigma':0.17}]],
            [['gaussmf',{'mean':6.,'sigma':7.6}],['gaussmf',{'mean':33.,'sigma':7.6}],['gaussmf',{'mean':60.,'sigma':8.5}]]]
mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS_CO(Xtrain, Ytrain, C, mfc)
anf.trainHybridJangOffLine(epochs=5)
Yp = anfis.predict(anf, Xtrain)
error_step1 = (Ytrain.reshape(len(Yp), 1)-Yp).transpose().dot(C).dot(Ytrain.reshape(len(Yp), 1)-Yp)
anf.plotErrors()

## step 2
Strain = t_train[:,3:5]
Y_step2 = Ytrain - Yp.ravel()
gp_step2 = GaussianProcess(theta0 = [2., 2.], thetaL = [1e-1, 1e-1], thetaU = [20., 20.])
gp_step2.fit(Strain, Y_step2)
C_step2 = gp_step2.reduced_likelihood_function(gp_step2.theta_)[1]['C']
C_step2 = numpy.dot(C_step2, C_step2.T)
predict_step2 = gp_step2.predict(Strain)
predict_step2 = predict_step2.reshape(len(Yp), 1)
error_ite0 = (Ytrain.reshape(len(Yp), 1)- Yp - predict_step2).transpose().dot(Ytrain.reshape(len(Yp), 1)- Yp - predict_step2)

## step 3
anf_step3 = anfis.ANFIS_CO(Xtrain, Ytrain, C_step2, mfc)
anf_step3.trainHybridJangOffLine(epochs=5)
Yp_step3 = anfis.predict(anf_step3, Xtrain)
error_step3 = (Ytrain.reshape(len(Yp_step3), 1)-Yp_step3).transpose().dot(C_step2).dot(Ytrain.reshape(len(Yp_step3), 1)-Yp_step3)
anf_step3.plotErrors()

## step 4 
Y_step4 = Ytrain - Yp_step3.ravel()
gp_step4 = GaussianProcess(theta0 = [2., 2.], thetaL = [1e-1, 1e-1], thetaU = [20., 20.])
gp_step4.fit(Strain, Y_step4)
C_step4 = gp_step2.reduced_likelihood_function(gp_step4.theta_)[1]['C']
C_step4 = numpy.dot(C_step4, C_step4.T)
predict_step4 = gp_step4.predict(Strain)
predict_step4 = predict_step4.reshape(len(Yp_step3), 1)
error_ite1 = (Ytrain.reshape(len(Yp_step3), 1) - Yp_step3 - predict_step4).transpose().dot(Ytrain.reshape(len(Yp_step3), 1) - Yp_step3 - predict_step4)

## step 5&6
Xtest = t_test[:,0:3]
Ytest = t_test[:,5]
Stest = t_test[:,3:5]
Y_pre = anfis.predict(anf_step3, Xtest) + gp_step4.predict(Stest).reshape(len(Ytest), 1)
error_test = (Ytest.reshape(len(Ytest), 1) - Y_pre).transpose().dot(Ytest.reshape(len(Ytest),1) - Y_pre)
MSE = numpy.sum(error_test/len(Ytest))

## Plotting figure
print('Prediction plot using anfis_co')
plt.scatter(Stest[:, 0], Stest[:, 1], c = Y_pre.ravel(), marker = 's')
# plt.gray()
plt.show()

print('True plot')
plt.scatter(Stest[:, 0], Stest[:, 1], c = Ytest, marker = 's')
# plt.gray()
plt.show()

if round(1000*MSE) == 262:
    print('Program ran correctly!')
else:
    print('Program ran with errors, please check!')


