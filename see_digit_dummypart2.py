import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
def main():
    data = pd.read_csv("C:/documents_local/`projet_math/data_csv/mnist_train.csv")
    data_testing_complete = pd.read_csv("C:/documents_local/`projet_math/data_csv/mnist_test.csv")

    data = np.array(data)
    label_train = data[:,0]
    image_train = np.delete(data, 0, 1)


    #Todo je pense que les poids devraint etre entre 0 et 1 et en ce moment ils sont plus grand que ca des fois
    #Todo Update -> j'ai remplace np.randn par np.random() (entre 0 et 1) - 0.5 comme ca c'est pas mal de -0.5 a 0.5 et c'est mieux
    w1 = np.random.random((15, 784)) - 0.5
    b1 = np.random.random((15, 1)) - 0.5
    w2 = np.random.random((10, 15)) - 0.5
    b2 = np.random.random((10, 1)) - 0.5


    mini_batch = 100
    epochs = 80
    learning_rate = 0.01

    for i in range(epochs):
        nb_correct = 0
        count = 1
        cost = 0
        w1average = np.zeros((15, 784))
        b1average = np.zeros((15, 1))
        w2average = np.zeros((10,15))
        b2average = np.zeros((10,1))

        for img, l in zip(image_train, label_train):

            answer = one_hot(l).T
            ima = img.reshape(784,1)

            z1 = np.dot(w1, ima) + b1
            a1 = np.tanh(z1)
            z2 = np.dot(w2, a1) + b2
            a2 = np.tanh(z2)


            if np.argmax(a2) == np.argmax(answer):
                nb_correct += 1

            #potenetiellement devoir ajouter un calcul pour la regularisatoin de la fonction cout
            #  lambda/2n * (sommation) w**2
            #  http://neuralnetworksanddeeplearning.com/chap3.html !!formule 87 trust!!
            
            #Backpropagation
            #Todo I guess que notre accuracy est poche parce que j<Ai mal fait quelque chose...(peut etre reviser ca)
            mse =  1/(2*count)*(np.subtract(a2, answer))**2
            deriv_error = 2 * np.subtract(a2, answer)
            cost += mse  #on a changé le cost a cause on a vu c'était pas la formule quadratique et la le cost partait a comme 26 et tendait vers 0 ish
            deriv_z2 = deriv_tanh(z2)
            partiel = np.multiply(deriv_error, deriv_z2)
            tempa = a1.T
            weights_update2_3 = np.dot(partiel, tempa)

            bias_update2_3 = partiel
            bias_update1_2 = np.multiply(np.dot(w2.T, bias_update2_3), deriv_tanh(z1))

            weight_update1_2 = np.dot(bias_update1_2, ima.T)


            w1average += weight_update1_2
            b1average += bias_update1_2
            w2average += weights_update2_3
            b2average += bias_update2_3




            if count % mini_batch == 0:
                w2 -= (learning_rate * w2average/count)
                b2 -= (learning_rate * b2average/count)
                w1 -= (learning_rate * w1average/count)
                b1 -= (learning_rate * b1average/count)

            count += 1

        print(f"Epoch #{i}: {nb_correct/600}  % accuracy, cost = {np.average(cost)}")


def deriv_tanh(matrice_a2):
    return (1 - (np.tanh(matrice_a2)**2))



def one_hot(y):
    reponse = np.zeros((1,10))
    reponse[np.arange(1), y] = 1
    return reponse

main()


