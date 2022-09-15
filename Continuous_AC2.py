import sys
sys.path.insert(0, 'Utilities')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, tb, X_f, layers, lb, ub):
        
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.lb = lb
        self.ub = ub
                      
        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.u0 = u0
        #self.v0 = v0
        


        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])
        
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.u0_pred,  _ = self.net_u(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf, self.t_lb_tf) 
        self.u_ub_pred, self.u_x_ub_pred = self.net_u(self.x_ub_tf, self.t_ub_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf) 
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred)) 
                   
        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate= 1.0e-4)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases): 
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y


    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        
        return u, u_x

    def net_f(self, x, t):
        u, u_x = self.net_u(x,t)
        
        u_t = tf.gradients(u, t)[0]  ###
        u_xx = tf.gradients(u_x, x)[0]

        f = 5.0*u - 5.0*u**3 + 0.0001*u_xx - u_t #####
        return f
    
    def callback(self, loss):
        global countnum
        print(str(countnum)+' - Loss in loop: %.3e' % (loss))
        countnum += 1
        #print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,   ##############
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 1 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
        
        tf_dict = {self.x0_tf: X_star[:,0:1], self.t0_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)  

        tf_dict = {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]}
        
        f_star = self.sess.run(self.f_pred, tf_dict)
       
        return u_star,  f_star
    
if __name__ == "__main__":     
    noise = 0.0      

    # Domain bounds
    lb = np.array([-1.0, 0.0]) #x,t
    ub = np.array([1.0, 0.1]) #x,t #####################

    N0 = 50
    N_b = 50
    N_f = 200 #20000
    layers = [2, 100, 100, 100, 100, 1]  
    
    data = scipy.io.loadmat('Data/AC2.mat')

    t = data['tt1'].flatten()[:,None] #(1，201) (1,21)
    x = data['x'].flatten()[:,None] #(1，512)
    Exact = data['uu1'] ##########

    X, T = np.meshgrid(x,t) 
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) 
    u_star = Exact.T.flatten()[:,None]      ###########      

    idx_x = np.random.choice(x.shape[0], N0, replace=False) ####replace=False
    x0 = x[idx_x,:]
    u0 = Exact[idx_x,0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=True) ###replace=False
    tb = t[idx_t,:]

    X_f = lb + (ub-lb)*lhs(2, N_f)
        
    model = PhysicsInformedNN(x0, u0, tb, X_f, layers, lb, ub)

    countnum = 1
    start_time = time.time()                
    model.train(100) #10000
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred, f_pred = model.predict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))                     
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    F_pred = griddata(X_star, f_pred.flatten(), (X, T), method='cubic')


    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')

    ######## Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.1, right=0.9, wspace=0)  ###
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[5] * np.ones((2, 1)), line, 'k--', linewidth=1) ####################
    ax.plot(t[10] * np.ones((2, 1)), line, 'k--', linewidth=1) ######################
    ax.plot(t[15] * np.ones((2, 1)), line, 'k--', linewidth=1) ######################3

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc='best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|u(t,x)|$', fontsize=10)

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.8)  ####
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[:, 5], 'b-', linewidth=2, label='Exact') #################
    ax.plot(x, U_pred[5, :], 'r--', linewidth=2, label='Prediction') ################
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|u(t,x)|$')
    ax.set_title('$t = %.2f$' % (t[5]), fontsize=10) ##################
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])  ###################
    ax.set_ylim([-1.1, 0.5])  #####################

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[:, 10], 'b-', linewidth=2, label='Exact') ########################
    ax.plot(x, U_pred[10, :], 'r--', linewidth=2, label='Prediction') ######################
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|u(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])  #########################
    ax.set_ylim([-1.1, 0.5])  #################
    ax.set_title('$t = %.2f$' % (t[10]), fontsize=10) ####################3
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[:, 15], 'b-', linewidth=2, label='Exact') ##################
    ax.plot(x, U_pred[15, :], 'r--', linewidth=2, label='Prediction') ###################
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|u(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])  ####################
    ax.set_ylim([-1.1, 0.5])  ####################
    ax.set_title('$t = %.2f$' % (t[15]), fontsize=10) #################



