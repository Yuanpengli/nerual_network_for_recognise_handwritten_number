__author__ = 'liyuanpeng'

import numpy
import math
from numpy import *
import random
import struct
import matplotlib.pyplot as plt

# def sigmoid(z):
#     return 1.0/(1.0+numpy.exp(-z))

class network(object):

    def __init__(self,sizes):

        self.num_layers = len(sizes)

        self.sizes = sizes
        # ''' it is tested all right'''
        #
        # self.biases  = []
        # self.biases.append(mat([-0.4,0.2]).T)
        # self.biases.append(mat([0.1]))
        #
        # self.weights = []
        # self.weights.append(mat([[0.2,-0.3],[0.4,0.1],[-0.5,0.2]]))
        # self.weights.append(mat([[-0.3],[-0.2]]))


        self.biases  = []

        for i in range(self.num_layers-1):
            self.biases.append(numpy.random.rand(sizes[i+1],1)-0.5)

        self.weights = []

        for i in range(self.num_layers-1):

            self.weights.append(numpy.random.rand(sizes[i],sizes[i+1])-0.5)


#
    def sigmoid(self,z):
        return 1.0/(1.0+exp(-z))

        #return .5 * (1 + numpy.tanh(.5 * z))



    def forward(self,a,i):

        #print '1111111111111111'


        a= mat(self.weights[i]).T*a+self.biases[i]

        #print a

        #print a

        a = self.sigmoid(a)

        #print a

        #for b in self:

        return a

    #def backward(self,input,o,final_o,i,y,learning_rate):
    def backward(self,c,i,output_vector,learning_rate):

        partical_o = [x*y for x,y in zip(c[i],(1.0-mat(c[i]))) ]

        if i ==len(self.sizes)-1 :

            partical_o_last = [x*y for x,y in zip(partical_o,mat(output_vector)-mat(c[i])) ]


            real_partical = []

            # print len(partical_o_last)

            for j in range(len(partical_o_last)):
                real_partical.append(numpy.asarray(partical_o_last)[j][0][0])

            partical_o_last=mat(real_partical).T

            # print partical_o_last
            #
            # print '11111111111'
            #
            # print c[i-1].T


            # partical_o_last_1.append(x for x in numpy.asarray(partical_o_last))
            #
            # print partical_o_last_1

            #partical_o_last = partical_o*(mat(y)-mat(o))   # there is a mistake , which is two vector paralell multiple to each other

            self.weights[i-1] += (learning_rate *partical_o_last *c[i-1].T).T  #there is a mistake which need to be solved, is (y-o).*o.*(1-o).

            self.biases[i-1] += learning_rate *partical_o_last


            return self
            #
            # print '222222'
            #
            # print self.weights[i-1]
            # print self.biases[i-1]


        elif i == len(self.sizes)-2 :

            sum_k =  [x*y for x,y in zip(c[i+1],(1.0-c[i+1]))]

            # print '222222222'
            # print sum_k
            # print mat(output_vector).T
            # print c[i+1]
            # print mat(output_vector).T-c[i+1]


            sum_k =  [x*y for x,y in zip(sum_k,mat(output_vector).T-c[i+1])]

            sum_partical = (mat(asarray(sum_k))*self.weights[i].T)

            #print sum_partical[0]*partical_o[0]


            replace1 =[x*y for x,y in zip(partical_o,sum_partical)]

            #print '111111'

            #replace1 = replace1
            #print mat(asarray(replace1)).T


            partical_o = mat(asarray(replace1)).T*mat(c[i-1]).T

            #print partical_o

            #print partical_o
            #print partical_o
            self.weights[i-1] += learning_rate *partical_o.T
            self.biases[i-1] +=learning_rate *mat(asarray(replace1)).T


        #return self.weights[i-1],self.biases[i-1]



    def SGD(self,sub_dataset):

        num_image = len(sub_dataset)

        before_sum=100.0


        # while True:
        #
        #
        sum = 0.0

        for j in range(num_image):

            num_layers = len(self.sizes)

            data = sub_dataset[j]


            result=data[0]


            c = []

            for i in range(num_layers-1):

                #print mat(result)

                c.append(result)

                #print result

                result=network.forward(self,result,i) # conclulate the y(x) in network


            #print mat(result)

            c.append(result)

            #print c

            y = sub_dataset[j][1]
                #print y

            for i in [num_layers-1,1]:

                t= network.backward(self,c,i,y,learning_rate=3.0)

        return self

    def test_result(self,data):

        input = data[0]
        #print data[0]

        for i in range(2):
            input = network.forward(self,input,i)

        return input




            #print sum

def collect_dataset(filename_1,filename_2):

    binfile = open(filename_1 , 'rb')
    buf = binfile.read()

    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)

    print numImages


    index += struct.calcsize('>IIII')

    input = struct.unpack_from('>47040000B' ,buf, index)
    #input = struct.unpack_from('>7840000B' ,buf, index)
    input = input[6271:-1]
    input = numpy.array(input)

    binfile = open(filename_2 , 'rb')
    buf = binfile.read()

    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)

    index += struct.calcsize('>IIII')

    #output = struct.unpack_from('>59992B' ,buf, index)
    output = struct.unpack_from('>59992B' ,buf, index)
    output = numpy.array(output)

    output_vector = []

    for i in range(len(output)):
        zeros_vector = zeros(10)
        zeros_vector[output[i]] = 1.0

        output_vector.append(zeros_vector)

    num_images = len(input)/(28*28)
    dataset =[]


    for i in range(num_images):

        dataset.append([mat(input[784*i:784*(i+1)]).T,list(output_vector[i])])

    return dataset






if __name__ == '__main__':
    # network_1 = network([2,3,4])
    # print network_1.weights
    #
    # print network_1.biases
    # a=mat([9.0,8.0]).T
    #
    # network_2 = network_1.SGD([[a,[1,1,0,0]]])





    #
    # print network_1.weights
    # print network_1.biases

    # network_1 = network([2,3,4])
    # a=mat([9.0,8.0]).T
    # tol=0.00001
    # network_2 = network_1.SGD([[a,[1,1,0,0]]],tol)




    #
    # network_1 = network([3,2,1])
    # a=mat([1.0,0.0,1.0]).T
    #
    # network_1 = network_1.SGD([[a,[1.0]]])
    #
    # print network_1.weights
    # print network_1.biases






    #print network_1.weights
    #
    # filename_1 = 't10k-images-idx3-ubyte'
    #
    #
    # filename_2 = 't10k-labels-idx1-ubyte'







    filename_1 = 'train-images-idx3-ubyte'


    filename_2 = 'train-labels-idx1-ubyte'



    dataset = collect_dataset(filename_1,filename_2)

    #dataset1 = dataset[5990:-1]
    #dataset1 =[dataset[0]]

    real_network = network([28*28,20,10])
    # print real_network.weights
    # print real_network.biases


    real_network= real_network.SGD(dataset)

    # print real_network.weights
    # print real_network.biases

    for i in range(6):

        output =real_network.test_result(dataset[i])
        print '1111111111'

        # print real_network.weights
        # print real_network.biases
        print output

        image = dataset[i][0].reshape(28,28)
        print dataset[i][1]

        fig = plt.figure()
        plotwindow = fig.add_subplot((230+(i+1)))
        plt.imshow(image , cmap='gray')
    plt.show()
    #
