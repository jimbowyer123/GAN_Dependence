# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:42:41 2018

@author: jb2968
"""

import numpy as np
import tensorflow as tf
from scipy.stats import chi2
import matplotlib.pyplot as plt
from HSIC import HSIC


# Need so define some hyper parameters for the model
disc_lr=0.01
gen_lr=0.0001
space_cyc_coef=2
latent_cyc_coef=2

# Need to define parameters determining the structure of our model
real_dim=100
latent_dim=10
d_hidden=[256]
g_hidden=[256]
e_hidden=[256]


epochs=100000
batch_size=400

# Need to build our generators, discriminators and encoders

def gen_xy(x,z,reuse=False):
    with tf.variable_scope('gen_xy',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':tf.concat([x,z],1)}
        
        for i in range(len(g_hidden)):
            
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],g_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(g_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(g_hidden))],real_dim,kernel_initializer=w_init)
        
        return flow['dense{}'.format(len(g_hidden)+1)]
    
def gen_yx(y,z,reuse=False):
    with tf.variable_scope('gen_yx',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':tf.concat([y,z],1)}
        
        for i in range(len(g_hidden)):
            
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],g_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(g_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(g_hidden))],real_dim,kernel_initializer=w_init)
        
        return flow['dense{}'.format(len(g_hidden)+1)]
    
def disc_x(x,reuse=False):
    
    with tf.variable_scope('disc_x',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':x}
        
        for i in range(len(d_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],d_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.leaky_relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(d_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(d_hidden))],1,kernel_initializer=w_init)
        flow['out']=tf.nn.sigmoid(flow['dense{}'.format(len(d_hidden)+1)])
        
        return flow['out']
    
def disc_y(y,reuse=False):
    
    with tf.variable_scope('disc_y',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':y}
        
        for i in range(len(d_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],d_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.leaky_relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(d_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(d_hidden))],1,kernel_initializer=w_init)
        flow['out']=tf.nn.sigmoid(flow['dense{}'.format(len(d_hidden)+1)])
        
        return flow['out']
    
def disc_z1(z1,reuse=False):
    
    with tf.variable_scope('disc_z1',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':z1}
        
        for i in range(len(d_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],d_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.leaky_relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(d_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(d_hidden))],1,kernel_initializer=w_init)
        flow['out']=tf.nn.sigmoid(flow['dense{}'.format(len(d_hidden)+1)])
        
        return flow['out']
    
def disc_z2(z2,reuse=False):
    
    with tf.variable_scope('disc_z2',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':z2}
        
        for i in range(len(d_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],d_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.leaky_relu(flow['dense{}'.format(i+1)])
            
        flow['dense{}'.format(len(d_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(d_hidden))],1,kernel_initializer=w_init)
        flow['out']=tf.nn.sigmoid(flow['dense{}'.format(len(d_hidden)+1)])
        
        return flow['out']
    
def enc_z1(x,y,reuse=False):
    with tf.variable_scope('enc_z1',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':tf.concat([x,y],1)}
        
        for i in range(len(e_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],e_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.relu(flow['dense{}'.format(i+1)])
            
        
        flow['dense{}'.format(len(e_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(e_hidden))],latent_dim,kernel_initializer=w_init)
        
        return flow['dense{}'.format(len(e_hidden)+1)]
    
def enc_z2(x,y,reuse=False):
    with tf.variable_scope('enc_z2',reuse=reuse):
        w_init=tf.contrib.layers.xavier_initializer()
        
        flow={'activ0':tf.concat([x,y],1)}
        
        for i in range(len(e_hidden)):
            flow['dense{}'.format(i+1)]=tf.layers.dense(flow['activ{}'.format(i)],e_hidden[i],kernel_initializer=w_init)
            flow['activ{}'.format(i+1)]=tf.nn.relu(flow['dense{}'.format(i+1)])
            
        
        flow['dense{}'.format(len(e_hidden)+1)]=tf.layers.dense(flow['activ{}'.format(len(e_hidden))],latent_dim,kernel_initializer=w_init)
        
        return flow['dense{}'.format(len(e_hidden)+1)]
    


X=tf.placeholder(tf.float32,shape=(None,real_dim))
Y=tf.placeholder(tf.float32,shape=(None,real_dim))
Z_1=tf.placeholder(tf.float32,shape=(None,latent_dim))
Z_2=tf.placeholder(tf.float32,shape=(None,latent_dim))

Gen_Y=gen_xy(X,Z_1)
Gen_X=gen_yx(Y,Z_2)

Enc_Z1=enc_z1(Gen_X,Y)
Enc_Z2=enc_z2(X,Gen_Y)

Recon_X=gen_yx(Gen_Y,Enc_Z2,reuse=True)
Recon_Y=gen_xy(Gen_X,Enc_Z1,reuse=True)

Recon_Z1=enc_z1(X,Gen_Y,reuse=True)
Recon_Z2=enc_z2(Gen_X,Y,reuse=True)

Disc_X_Real=disc_x(X)
Disc_X_Fake=disc_x(Gen_X,reuse=True)

Disc_Y_Real=disc_y(Y)
Disc_Y_Fake=disc_y(Gen_Y,reuse=True)

Disc_Z1_Real=disc_z1(Z_1)
Disc_Z1_Fake=disc_z1(Enc_Z1,reuse=True)

Disc_Z2_Real=disc_z2(Z_2)
Disc_Z2_Fake=disc_z2(Enc_Z2,reuse=True)


Disc_X_Loss=tf.reduce_mean(tf.log(Disc_X_Real)+tf.log(1.-Disc_X_Fake))
Disc_Y_Loss=tf.reduce_mean(tf.log(Disc_Y_Real)+tf.log(1.-Disc_Y_Fake))
Disc_Z1_Loss=tf.reduce_mean(tf.log(Disc_Z1_Real)+tf.log(1.-Disc_Z1_Fake))
Disc_Z2_Loss=tf.reduce_mean(tf.log(Disc_Z2_Real)+tf.log(1.-Disc_Z2_Fake))







