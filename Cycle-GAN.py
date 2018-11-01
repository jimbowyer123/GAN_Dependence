# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
from scipy.stats import chi2
import matplotlib.pyplot as plt
from HSIC import HSIC


# Coefficient weighting the cycle consistency loss
cycle_coef=2
disc_lr=0.01
gen_lr=0.0001

# Dimensions of our 2 distributions
dim=100
g_hidden=[256]
d_hidden=[256]

epochs=100000
batch_size = 400

# Define the two generators for our models with number of hidden layers defined by the g_hidden variable.

def gen_y(x,reuse=False):
    with tf.variable_scope('gen_y',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'relu0':x}
        
        for i in range(len(g_hidden)):
            
            # Hidden layers consist of a dense layer followed by a ReLu activation
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['relu{}'.format(i)],g_hidden[i],kernel_initializer=w_init)
            flow['relu{}'.format(i+1)]=tf.nn.relu(flow['dense{}'.format(i+1)])
        
        flow['dense{}'.format(len(g_hidden)+1)] = tf.layers.dense(flow['relu{}'.format(len(g_hidden))],dim,kernel_initializer=w_init)
        #flow['out']=tf.scalar_mul(tf.constant(2,tf.float32),tf.nn.tanh(flow['dense{}'.format(len(g_hidden)+1)]))
        print(flow.keys())
        return flow['dense{}'.format(len(g_hidden)+1)]
    
def gen_x(y,reuse=False):
    with tf.variable_scope('gen_x',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'relu0':y}
        
        for i in range(len(g_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['relu{}'.format(i)],g_hidden[i],kernel_initializer=w_init)
            flow['relu{}'.format(i+1)]=tf.nn.relu(flow['dense{}'.format(i+1)])
        
        flow['dense{}'.format(len(g_hidden)+1)] = tf.layers.dense(flow['relu{}'.format(len(g_hidden))],dim,kernel_initializer=w_init)
        #print(flow['dense{}'.format(len(g_hidden)+1)])
        #flow['out']=tf.nn.tanh(flow['dense{}'.format(len(g_hidden)+1)])
        
        return flow['dense{}'.format(len(g_hidden)+1)]
    
    # Define the two discriminators we use in our model with hidden layers defined by the d_hidden variable

def disc_y(y,reuse=False):
    with tf.variable_scope('disc_y',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'lrelu0':y}
        
        for i in range(len(d_hidden)):
            # Here the hidden layers consist of dense layers followed by a leaky relu activation
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['lrelu{}'.format(i)],d_hidden[i],kernel_initializer=w_init)
            flow['lrelu{}'.format(i+1)]=tf.nn.leaky_relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(d_hidden)+1)]=tf.layers.dense(flow['lrelu{}'.format(len(d_hidden))],1,kernel_initializer=w_init)
        flow['out']=tf.nn.sigmoid(flow['dense{}'.format(len(d_hidden)+1)])
        
        return flow['out']
    
def disc_x(x,reuse=False):
    with tf.variable_scope('disc_x',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'lrelu0':x}
        
        for i in range(len(d_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['lrelu{}'.format(i)],d_hidden[i],kernel_initializer=w_init)
            flow['lrelu{}'.format(i+1)]=tf.nn.leaky_relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(d_hidden)+1)]=tf.layers.dense(flow['lrelu{}'.format(len(d_hidden))],1,kernel_initializer=w_init)
        flow['out']=tf.nn.sigmoid(flow['dense{}'.format(len(d_hidden)+1)])
        
        return flow['out']
    
X=tf.placeholder(tf.float32,shape=(None,dim))
Y=tf.placeholder(tf.float32,shape=(None,dim))
    
X_gen=gen_x(Y)
Y_gen=gen_y(X)

print(X_gen)
print(Y_gen)

X_recon=gen_x(Y_gen,reuse=True)
print(X_recon)

Y_recon=gen_y(X_gen,reuse=True)

Disc_Y_true=disc_y(Y)
Disc_Y_fake=disc_y(Y_gen,reuse=True)

Disc_X_true=disc_x(X)
Disc_X_fake=disc_x(X_gen,reuse=True)


D_X_loss=-tf.reduce_mean(tf.log(Disc_X_true)+tf.log(1-Disc_X_fake))
D_Y_loss=-tf.reduce_mean(tf.log(Disc_Y_true)+tf.log(1-Disc_Y_fake))

G_X_loss=-tf.reduce_mean(tf.log(Disc_X_fake))
G_Y_loss=-tf.reduce_mean(tf.log(Disc_Y_fake))

X_Cyc_loss=tf.losses.absolute_difference(X,X_recon)
Y_Cyc_loss=tf.losses.absolute_difference(Y,Y_recon)

Cyc_loss=X_Cyc_loss+Y_Cyc_loss


D_loss=D_X_loss+D_Y_loss
G_loss=G_X_loss+G_Y_loss+cycle_coef*Cyc_loss



T_vars=tf.trainable_variables()

D_X_vars=[var for var in T_vars if var.name.startswith('disc_x')]
D_Y_vars=[var for var in T_vars if var.name.startswith('disc_y')]

G_X_vars=[var for var in T_vars if var.name.startswith('gen_x')]
G_Y_vars=[var for var in T_vars if var.name.startswith('gen_y')]

D_vars=D_X_vars+D_Y_vars
G_vars=G_X_vars+G_Y_vars


D_Optimizer=tf.train.GradientDescentOptimizer(disc_lr).minimize(D_loss,var_list=D_vars)
G_Optimizer=tf.train.AdamOptimizer(learning_rate=gen_lr).minimize(G_loss,var_list=G_vars)


    
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    #X_=np.load('C:/Users/Jimbowyer123/Documents/GitHub/GAN_Dependence/Data/X_data.npy')
    #Y_=np.load('C:/Users/Jimbowyer123/Documents/GitHub/GAN_Dependence/Data/Y_dependent.npy')
    
    for epoch in range(epochs):
        X_= np.random.normal(0,1,size=(100000,dim))
        Y_= 2*X_
        Dxl=0
        Dyl=0
        Gxl=0
        Gyl=0
        Cl=0
        Dl=0
        Gl=0
        for batch in range(int(X_.shape[0]/(batch_size))):
            X_batch1=X_[batch*batch_size:batch*batch_size+int(batch_size/2),:]
            Y_batch1=Y_[batch*batch_size:batch*batch_size+int(batch_size/2),:]
            X_batch2=X_[batch*batch_size+int(batch_size/2):(batch+1)*batch_size,:]
            Y_batch2=Y_[batch*batch_size+int(batch_size/2):(batch+1)*batch_size,:]
        
        
            
            #print(X_batch1.shape)
            #print(Y_batch1.shape)
            
            
            _,dxl,dyl,gxl,gyl,cl,dl,gl=sess.run([D_Optimizer,D_X_loss,D_Y_loss,G_X_loss,G_Y_loss,Cyc_loss,D_loss,G_loss],feed_dict={X:X_batch1,Y:Y_batch1})
            Dxl+=dxl
            Dyl+=dyl
            Gxl+=gxl
            Gyl+=gyl
            Cl+=cl
            Dl+=dl
            Gl+=gl
            
            _ = sess.run(G_Optimizer,feed_dict={X:X_batch2,Y:Y_batch2})
            
        if epoch%10==0:
            X_test=np.random.normal(0,1,size=(1000,dim))
            Y_test=2*X_test
            
            X_test_gen,Y_test_gen=sess.run([X_gen,Y_gen],feed_dict={Y:Y_test,X:X_test})
                                            
            Y_test_gen_normalized=Y_test_gen/2
            
            
            distances_x=np.zeros((1000,))
            distances_y=np.zeros((1000,))
            for i in range(1000):
                point_x=X_test_gen[i,:]
                point_y=Y_test_gen_normalized[i,:]
                #print(point)
                distances_x[i]=np.dot(point_x,point_x)
                distances_y[i]=np.dot(point_y,point_y)
            #print(distances)
            #m_distances=np.sqrt(distances)
            #print(m_distances)
            ordered_distances_x=np.sort(distances_x)
            ordered_distances_y=np.sort(distances_y)

            quantiles=np.linspace(1,1000,num=1000)
            quantiles=(quantiles-0.5)/1000
            chi=chi2
            quants=chi.ppf(quantiles,df=100)
            plt.plot(quants,ordered_distances_x)
            plt.xlim(50,200)
            plt.ylim(50,200)
            plt.title('Generate_X')
            plt.show()
            
            plt.plot(quants,ordered_distances_y)
            plt.xlim(50,200)
            plt.ylim(50,200)
            plt.title('Generate_Y')
            plt.show()
            
        if epoch%100==0:
            X_test=np.random.normal(size=(1000,dim))
            Y_test=2*X_test
            
            x_gen , y_gen = sess.run([X_gen,Y_gen],feed_dict={Y:Y_test,X:X_test})
            
            print(' ')
            print('X_gen HSIC: {}'.format(HSIC(x_gen,Y_test)))
            print('Y_gen HSIC: {}'.format(HSIC(X_test,y_gen)))
            print('True HSIC: {}'.format(HSIC(X_test,Y_test)))
        
        print('\nEpoch: {}'.format(epoch))
        print('Discriminator_X_Loss: {}'.format(Dxl))
        print('Discriminator_Y_Loss: {}'.format(Dyl))
        print('Generator_X_Loss: {}'.format(Gxl))
        print('Generator_Y_Loss: {}'.format(Gyl))
        print('Cycle_Loss: {}'.format(Cl))
        print('Discriminator_Loss: {}'.format(Dl))
        print('Generator_Loss: {}'.format(Gl))


        
        
    
            