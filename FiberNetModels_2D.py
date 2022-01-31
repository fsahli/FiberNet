#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Implementation of the FiberNet model to a 2D case
Author: Carlos Ruiz Herrera, Thomas Grandits, Paris Perdikaris, Francisco Sahli Costabal, Simone Pezzuto
"""

import time
import numpy as np
import tensorflow as tf
from pyDOE import lhs
from fimpy.solver import FIMPY
from scipy.spatial import cKDTree
from collections import defaultdict
from mesh_tools import cellToPointData, pointToCellData
from math_tools import eigenDecompProd, matMulProdSum, metricNormMatrix
from tqdm.auto import trange

# Set up tensorflow in graph mode
tf.compat.v1.disable_eager_execution()

# PINN class construction

class MultiAnisoEikonalPINN_2D:
  # Initialize the class
  def __init__(self, X, triangs, parallel, X_e, T_e, layers, CVlayers, Tmax,
               CVmax=1.0, lambda_df=1., lambda_pde=1e-4,
               lambda_tve=1e-2, lambda_tva=1e-9, 
               jobs=4):

    # Basic variables
    points = X   # Just to distinguish geometric and NN calculations
    # Creation of smooth basis mesh

    # Check measurement points are subset of the collocation points
    self.kdtree_X = cKDTree(points)

    # Creation of Manifold Basis for vertices
    p1 = np.concatenate([np.ones([triangs.shape[0],1]),np.zeros([triangs.shape[0],2])], axis=-1)
    p2 = np.concatenate([np.zeros([triangs.shape[0],1]),np.ones([triangs.shape[0],1]),np.zeros([triangs.shape[0],1])], axis=-1)
    p3 = np.concatenate([np.zeros([triangs.shape[0],2]),np.ones([triangs.shape[0],1])], axis=-1)
    P = np.concatenate([p1,p2,p3],axis=-1) #geom.createLocalManifoldBasis(X[triangs], smooth_basis)
    self.P_p = cellToPointData(points, triangs, P.reshape([-1, 9])).reshape([-1, 3, 3]).astype(np.float32)
    
    # Assign class parameters
    self.X = X
    self.p_NN = parallel
    self.Tmax = Tmax
    self.lb = X.min(0)
    self.ub = X.max(0)+[0.,0.,1.]
    self.T_e = T_e
    self.X_e = X_e
    self.layers = layers
    self.CVlayers = CVlayers
    self.points = points
    self.triangs = triangs

    # Initialize NN
    weights = []
    biases = []
    for i in np.arange(self.p_NN):
      w, b = self.initialize_NN(layers)
      weights.append(w)
      biases.append(b)
    self.weights = weights
    self.biases = biases
    self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)

    # Assign tf constants
    self.C = tf.constant(CVmax, dtype=tf.float32)
    self.alpha_e = tf.constant(lambda_tve, dtype=tf.float32)
    self.alpha = tf.constant(lambda_tva, dtype=tf.float32)
    self.lambda_DF = tf.constant(lambda_df, dtype=tf.float32)
    self.lambda_PDE = tf.constant(lambda_pde, dtype=tf.float32)

    # tf placeholders and graph
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      intra_op_parallelism_threads=jobs,
                                      inter_op_parallelism_threads=jobs,
                                      device_count={'CPU': jobs})
    config.gpu_options.allow_growth = True
    self.sess = tf.compat.v1.Session(config=config)

    self.X_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.X.shape[1]])
    self.P_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.P_p.shape[1], self.P_p.shape[2]])
    self.T_e_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.T_e.shape[1]])
    self.X_e_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.X_e.shape[1], None])

    self.T_pred, self.CV_pred, self.f_T_pred, self.eV_TV_func, self.aV_TV_func = self.net_eikonal(self.X_tf,
                                                                                                  self.P_p_tf)                                
    self.T_e_pred = self.net_data(self.X_e_tf)
    self.pde_loss = self.lambda_PDE * tf.reduce_mean(tf.square(self.f_T_pred))
    self.tv_loss = self.alpha_e * tf.reduce_mean(self.eV_TV_func) + self.alpha * tf.reduce_mean(self.aV_TV_func)
    self.data_fidelity_loss = self.lambda_DF * tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred))

    self.loss = self.data_fidelity_loss + self.pde_loss + self.tv_loss

    # Define optimizer (ADAM)
    self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
    self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

    # Initialize Tensorflow variables
    init = tf.compat.v1.global_variables_initializer()
    self.sess.run(init)

  # Initialize network weights and biases using Xavier initialization
  def initialize_NN(self, layers):
    # Xavier initialization
    def xavier_init(size):
      in_dim = size[0]
      out_dim = size[1]
      xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
      return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)

    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
      W = xavier_init(size=[layers[l], layers[l + 1]])
      b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
      weights.append(W)
      biases.append(b)
    return weights, biases

  # Construct neural network (Forward Propagation)
  def neural_net(self, X, weights, biases):
    num_layers = len(weights) + 1

    H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
    for l in range(0, num_layers - 2):
      W = weights[l]
      b = biases[l]
      H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

  # TV-Huber regularization function 
  def TVHuber(self, nabla_x, huber_norm_eps):
    nabla_x_norm_squared = tf.reduce_sum(nabla_x**2, axis=-1, keepdims=True)
    nabla_x_norm = tf.sqrt(nabla_x_norm_squared)
    nabla_x_reg_term = tf.where(nabla_x_norm <= huber_norm_eps,
               0.5/huber_norm_eps * nabla_x_norm_squared,
               (tf.sqrt(tf.maximum(nabla_x_norm_squared, huber_norm_eps**2))
                   - 0.5 * huber_norm_eps))

    return nabla_x_reg_term, nabla_x_norm_squared, nabla_x_norm

# Application of Multimap Anistropic Eikonal equation and Huber Regularizations
  def net_eikonal(self, X, P_p_loc, eps=1.e-9):
    C = self.C
    T = []
    T_x = []
    for i in np.arange(self.p_NN):
      T.append(self.neural_net(X, self.weights[i], self.biases[i]))
      T_x.append(tf.gradients(T[i], X)[0])
    T = tf.concat(T,-1)
    CV = self.neural_net(X, self.CVweights, self.CVbiases)
    eV = C * (tf.sigmoid(CV[:,:2]))
    aV = tf.tanh(CV[:,2])
    self.CV = CV
    self.evals = eV

    T_x = tf.concat(T_x,-1)
    aV_x = tf.gradients(aV, X)[0]
    eV_x = tf.concat([tf.gradients(eV[:,0], X)[0],tf.gradients(eV[:,1], X)[0]],axis=-1)
    self.CV_x = [eV_x, aV_x]    

    eV_flat = tf.cast(tf.reshape(eV, [-1]), dtype=tf.float64)
    aV_flat = tf.cast(tf.reshape(aV, [-1]), dtype=tf.float64)
    zero_e = tf.zeros_like(eV_flat[0::2])
    aVr = tf.sqrt(tf.maximum(1-aV_flat**2,eps))
    eVM_mat = tf.reshape(tf.stack([eV_flat[0::2], zero_e, zero_e, eV_flat[1::2]], axis=-1), [-1, 2, 2])
    aVM_mat = tf.reshape(tf.stack([aV_flat, -1.*aVr,aVr, aV_flat], axis=-1), [-1, 2, 2])

    D = eigenDecompProd(aVM_mat, eVM_mat)
    self.D = D
 
    P_p_local = tf.cast(P_p_loc, dtype=np.float64)

    zeros = tf.zeros_like(aVM_mat[..., 0, 0])
    ones = tf.ones_like(aVM_mat[..., 0, 0])
    aVM_3D = tf.reshape(tf.stack([aVM_mat[..., 0, 0], aVM_mat[..., 0, 1], zeros,
                                  aVM_mat[..., 1, 0], aVM_mat[..., 1, 1], zeros,
                                  zeros, zeros, ones], axis=-1), [-1, 3, 3])
    evecs = matMulProdSum(P_p_local, aVM_3D)
    self.evecs = tf.cast(evecs, dtype=tf.float32)

    evals3D = tf.reshape(tf.stack([eVM_mat[..., 0, 0], zeros, zeros,
                                    zeros, eVM_mat[..., 1, 1], zeros,
                                    zeros, zeros, zeros], axis=-1), [-1, 3, 3])


    D_canon_3D = eigenDecompProd(evecs, evals3D)
    D_canon_3D = tf.cast(D_canon_3D, dtype=tf.float32)
    self.D_canon_3D = D_canon_3D

    # Eikonal Residuals
    eik_loss = []
    for i in np.arange(self.p_NN):
      eik_loss.append(self.Tmax[i]*metricNormMatrix(D_canon_3D, T_x[...,3*i:3*i+3], ret_sqrt=True) - 1)
    eik_loss = tf.transpose(tf.stack(eik_loss,0))

    # Huber Regularization
    self.nabla_eV_reg_term = self.TVHuber(eV_x, 1e-3)[0]
    self.nabla_aV_reg_term = self.TVHuber(aV_x, 1e-3)[0]

    return (T, CV, eik_loss, self.nabla_eV_reg_term, self.nabla_aV_reg_term)

  def net_data(self, X_e):
    T_e = []
    for i in np.arange(self.p_NN):
      T_e.append(self.neural_net(X_e[...,i], self.weights[i], self.biases[i]))
    T_e = tf.concat(T_e,-1)

    return T_e

  def callback(self, loss):
    self.lossit.append([loss])
    # print('Loss: %.3e, DF: %.3e, PDE: %.3e' % (loss[0],loss[1],loss[2]))

  def train_Adam_minibatch(self, nEpoch, size=50):

    self.lossit = []

    start_time = time.time()
    idx_global = np.arange(self.X.shape[0])
    np.random.shuffle(idx_global) 
    splits = np.array_split(idx_global, idx_global.shape[0] // size)
    pbar = trange(nEpoch,desc='Training')
    for ep in pbar:
      for it, idx in enumerate(splits):
        tf_dict = {self.X_tf: self.X[idx],
                   self.X_e_tf: self.X_e,
                   self.T_e_tf: self.T_e,
                   self.P_p_tf: self.P_p[idx]}
        self.sess.run(self.train_op_Adam, tf_dict)

      loss_value = self.sess.run(self.loss, tf_dict)
      loss_df, loss_pde = self.sess.run((self.data_fidelity_loss, self.pde_loss), tf_dict)
      elapsed = time.time() - start_time
      pbar.set_postfix_str('Loss: %.3e, DF: %.3e, PDE: %.3e, Time: %.2f' %
                           (loss_value, loss_df, loss_pde, elapsed))
      self.lossit.append([loss_value, loss_df, loss_pde])
      start_time = time.time()

    pbar.close()

    return self.lossit

  def predict(self, X_star):

    indices = self.kdtree_X.query(X_star)[1]
    P_p_predict = self.P_p[indices]

    tf_dict = {self.X_tf: X_star,
               self.P_p_tf: P_p_predict}

    result = self.sess.run([self.Tmax*self.T_pred, self.CV_pred, self.CV_x, self.D, self.D_canon_3D,
                                              self.evals, self.evecs, self.f_T_pred], tf_dict) ##T_normalization: T out original scale

    return result

  def predict_errors(self):

    tf_dict = {self.X_tf: self.X,
               self.X_e_tf: self.X_e,
               self.T_e_tf: self.T_e,
               self.P_p_tf: self.P_p}

    total_loss, pde_loss, tv_loss = self.sess.run([self.loss, self.pde_loss, self.tv_loss], tf_dict)
    return total_loss, pde_loss, tv_loss

class SyntheticDataGenerator2D:
  """
  Create a set of cardiac activation maps on 2D grid mesh

  Parameters:
  grid_points: int number of points on the side of the square grid (total points = grid_points x grid_points)
  sample_points: int total number of points to sample across all maps (each map has sample_points/maps_number)
  maps_number: int number of activation maps desired
  """
  def __init__(self, grid_points=35, sample_points=245, maps_number=2, noise=0.) -> None:
    # Grid points must be more than sample points
    assert grid_points**2 > sample_points

    # Create mesh points         
    x = y = np.linspace(-1,1,grid_points)[:,None]
    X_m, Y_m = np.meshgrid(x,y)
    self.X_m, self.Y_m = X_m, Y_m
    X = X_m.flatten()[:,None]
    Y = Y_m.flatten()[:,None]
    Z = np.zeros_like(X)
    self.points = np.concatenate([X,Y,Z],axis=-1)

    # Create conduction velocity values
    cv = self.get_simulated_CV(X,Y)
    self.cv = cv
    eigenvectors = cv[0:4]
    eigenvectors.reshape(X_m.shape[0],X_m.shape[1],4)

    # Create mesh triangles
    triangles_list = []
    for i in np.arange(len(self.points)-grid_points):
      if i%grid_points!=grid_points-1:
        triangles_list.append([3,i,i+1,i+grid_points])
        triangles_list.append([3,i+1,i+grid_points,i+grid_points+1])
    triangles_array = np.array(triangles_list)
    self.triangles = triangles_array[:,1:]

    ## Create D tensor
    zed = np.zeros_like(cv[4])
    normal_cv = np.ones_like(cv[4]) * 1.e-3
    D_init = np.stack([cv[4],cv[6],zed,cv[6],cv[5],zed,zed,zed,normal_cv], axis=-1).reshape([-1,3,3])
    self.fiber_vecs = np.linalg.eigh(D_init)[1]
    D_init_cells = pointToCellData(self.points, self.triangles, D_init)

    # Select measurement points and initiation sites
    measurement_mask = np.zeros(self.points.shape[0], dtype=bool)
    kdtree_X = cKDTree(self.points)
    X_t = lhs(2, sample_points, 'c')*2-1
    X_train = np.concatenate([X_t,Z[:X_t.shape[0]]],axis=-1)
    sample_indices = kdtree_X.query(X_train)[1]
    sample_indices = self.remove_duplicates(sample_indices)
    sample_indices = sample_indices[:len(sample_indices)//maps_number*maps_number] #This might throw off exact number of sample points
    measurement_mask[sample_indices] = True
    print("Real number of sample points taken: ", len(sample_indices))
    initiation_sites = lhs(2, 5, 'm')*2-1
    initiation_sites = np.concatenate([initiation_sites,Z[:initiation_sites.shape[0]]],axis=-1)
    initiation_sites = kdtree_X.query(initiation_sites)[1]

    # Get activation times from initiation sites and cv with FIM method
    split_sample_points = np.array_split(sample_indices, maps_number)
    fim = FIMPY.create_fim_solver(self.points, self.triangles, D_init_cells, device='cpu', use_active_list=False)
    X_dirichlet = []
    phi_dirichlet = []
    phi_max = []
    phis = []
    x0_vals = np.zeros(maps_number)
    for i, idx in enumerate(split_sample_points):
      phi = fim.comp_fim(initiation_sites[i], x0_vals[i])
      phi += noise * np.random.randn(phi.shape[0])
      phis.append(phi)
      m_mask = np.zeros(self.points.shape[0], dtype=bool)
      m_mask[idx] = True
      X_dirichlet.append(self.points[m_mask])
      phi_dirichlet.append(phi[m_mask])
      phi_max.append(phi_dirichlet[i].max())
      phi_dirichlet[i] = phi_dirichlet[i]/phi_max[i]
    X_dirichlet = np.transpose(np.stack(X_dirichlet, axis=0),[1,2,0])
    phi_dirichlet = np.transpose(np.stack(phi_dirichlet, axis=0))
    phis = np.transpose(np.stack(phis, axis=0))

    self.phi = phis
    self.phi_max = phi_max
    self.T_e = phi_dirichlet
    self.X_e = X_dirichlet

  def get_simulated_CV(self, X, Y):
    """
    Returns values of 2d FiberNet paper simulated conduction velocities in different formats
    The values are:
    [eigenvector1_x_val, eigenvector1_y_val, eigenvector2_x_val, eigenvector1_y_val,
    matrix_format_val_00, matrix_format_val_11, matrix_format_val_10]
    """
    mask = np.less_equal(np.sqrt((X+1)**2 + 2*(Y+1)**2),np.sqrt(2*(X-1)**2 + (Y-1)**2))
    d1 = mask*1.0 + ~mask*0.5
    d2 = mask*0.5 + ~mask*1.0
    d12 = mask*0.0 + ~mask*0.0
    c = (d1+d2)/2
    r = np.sqrt((d1-c)**2+d12**2)
    a = np.arctan2(d12,(d1-c))/2
    e1 = c+r
    e2 = c-r
    e1x = e1*np.cos(a)
    e1y = e1*np.sin(a)
    e2x = -e2*np.sin(a)
    e2y = e2*np.cos(a)
    return np.array([e1x, e1y, e2x, e2y, d1, d2, d12])
  
  def remove_duplicates(self, old_list):
    unwanted = []
    tally = defaultdict(list)
    for i,item in enumerate(old_list):
        tally[item].append(i)
    duples = ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)
    for duplicates in sorted(duples):
        dupe = duplicates[1][1:]
        for name in dupe: 
          unwanted.append(name)
    new_list = [i for j, i in enumerate(old_list) if j not in unwanted]
    return new_list

  def get_geometry(self):
    return self.points, self.triangles, self.X_m, self.Y_m

  def get_activation_maps(self):
    return self.phi, self.X_e, self.T_e, self.phi_max

  def get_fiber_vectors(self):
    return self.fiber_vecs, self.cv

def D_printer(D):
    d1 = D[:,0,0]
    d2 = D[:,1,1]
    d12 = D[:,0,1]
    c = (d1+d2)/2
    r = np.sqrt((d1-c)**2+d12**2)
    a = np.arctan2(d12,(d1-c))/2
    e1 = c+r
    e2 = c-r
    e1x = e1*np.cos(a)
    e1y = e1*np.sin(a)
    e2x = -e2*np.sin(a)
    e2y = e2*np.cos(a)
    return np.array([e1x, e1y, e2x, e2y])
