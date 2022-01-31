#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Implementation of the FiberNet model to a 3D case
Author: Carlos Ruiz Herrera, Thomas Grandits, Paris Perdikaris, Francisco Sahli Costabal, Simone Pezzuto
"""

import time
import numpy as np
import pyvista as pv
import vtk
import tensorflow as tf
from fimpy.solver import FIMPY
from scipy.spatial import cKDTree
from mesh_tools import calculateSurfaceNormalsManifold, createLocalManifoldBasis, cellToPointData
from math_tools import eigenDecompProd, matMulProdSum, metricNormMatrix
from tqdm.auto import trange

# Set up tensorflow in graph mode
tf.compat.v1.disable_eager_execution()

# PINN class construction

class MultiAnisoEikonalPINN_3D:
  # Initialize the class
  def __init__(self, X, triangs, parallel, X_e, T_e, ind, 
               layers, CVlayers, smooth_basis_file, 
               CVmax=1.0, lambda_df=1., lambda_pde=1e-4, 
               lambda_tve=1e-2, lambda_tva=1e-9, 
               jobs=4):
    
    # Basic variables
    points = X   # Just to distinguish geometric and NN calculations
    normals = calculateSurfaceNormalsManifold(points, triangs)

    # Creation of smooth basis mesh
    smooth_basis_mesh = pv.UnstructuredGrid(smooth_basis_file)
    smooth_basis = smooth_basis_mesh.cell_data["vf_smooth"]
    if not np.allclose(np.sum(smooth_basis * normals, axis=-1, keepdims=True),0.,atol=1e-4):
      smooth_basis = smooth_basis - normals * np.sum(smooth_basis * normals, axis=-1, keepdims=True)
      smooth_basis /= np.linalg.norm(smooth_basis, axis=-1, keepdims=True)

    # Check measurement points are subset of the collocation points
    self.kdtree_X = cKDTree(points)
    assert(np.allclose(self.kdtree_X.query(X_e)[0], 0.))

    # Check data and normalize time values
    assert X_e.shape[0]==T_e.shape[0]
    # assert X_e.shape[-1]==parallel and T_e.shape[-1]==parallel
    assert X_e.shape[1]==3 and len(T_e.shape)==2
    T_top = np.array([])
    T_base = np.zeros(parallel)
    x_range = (X.max(0)-X.min(0)).flatten()
    for i in range(parallel):
      t_range = T_e[ind[i]:ind[i+1],...].max(0)-T_e[ind[i]:ind[i+1],...].min(0)
      assert all(np.logical_and(1. < x_range, x_range < 1.e3)) # Check that spatial measurement units are mm
      assert all(np.logical_and(1. < t_range, t_range < 1.e3)) # Check time measurements are in ms and from a single cycle
      if T_e[ind[i]:ind[i+1],...].min(0).flatten() > 10. or T_e[ind[i]:ind[i+1],...].min(0).flatten() < 0.:
        T_base[i] = T_e.min(0)
        T_e[ind[i]:ind[i+1],...] -= T_base[i]
      T_top = np.append(T_top, T_e[ind[i]:ind[i+1],...].max(0))
      T_e[ind[i]:ind[i+1],...] /= T_top[i]

    # Creation of Manifold Basis for vertices
    P = createLocalManifoldBasis(X, triangs, smooth_basis)
    self.P_p = cellToPointData(points, triangs,
                                                P.reshape([-1, 9])).reshape([-1, 3, 3]).astype(np.float32)
    
    # Assign class parameters
    self.X = X
    self.p_NN = parallel
    self.Tmax = T_top
    self.Tmin = T_base
    self.lb = X.min(0)
    self.ub = X.max(0)
    self.normals = normals
    self.T_e = T_e
    self.X_e = X_e
    self.ind = ind
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
    self.X_e_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.X_e.shape[1]])
    self.ind_tf = tf.compat.v1.placeholder(tf.int32, shape=[None])

    self.T_pred, self.CV_pred, self.f_T_pred, self.eV_TV_func, self.aV_TV_func = self.net_eikonal(self.X_tf, 
                                                                                                  self.P_p_tf)                                
    self.T_e_pred = self.net_data(self.X_e_tf, self.ind_tf)
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

  def net_data(self, X_e, ind):
    T_e = []
    for i in np.arange(self.p_NN):
      T_e.append(self.neural_net(X_e[ind[i]:ind[i+1],...], self.weights[i], self.biases[i]))
    T_e = tf.concat(T_e,0)

    return T_e

  def callback(self, loss):
    self.lossit.append(loss)
    # print('Loss: %.5e (loss))

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
                self.P_p_tf: self.P_p[idx],
                self.ind_tf: self.ind}
        self.sess.run(self.train_op_Adam, tf_dict)

      loss_value = self.sess.run(self.loss, tf_dict)
      loss_df, loss_pde = self.sess.run((self.data_fidelity_loss, self.pde_loss), tf_dict)
      elapsed = time.time() - start_time
      #pbar.set_postfix({'Loss': loss_value, 'DF': loss_df, 'PDE': loss_pde, 'Time': elapsed})
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

    result = self.sess.run([self.Tmax*self.T_pred + self.Tmin, self.CV_pred, self.CV_x, self.D, self.D_canon_3D,
                                              self.evals, self.evecs, self.f_T_pred], tf_dict)

    return result

  def predict_errors(self):

    tf_dict = {self.X_tf: self.X,
               self.X_e_tf: self.X_e,
               self.T_e_tf: self.T_e,
               self.P_p_tf: self.P_p,
               self.ind_tf: self.ind}

    total_loss, df_loss, pde_loss, tv_loss = self.sess.run([self.loss, self.data_fidelity_loss,
                                                            self.pde_loss, self.tv_loss], tf_dict)
    return total_loss, df_loss, pde_loss, tv_loss

class SyntheticDataGenerator3D:
  """
  Create a set of cardiac activation maps from a geometry file with fiber orientations

  Parameters:
  vtk_file: geometry file in .vtk format with a cell data field called "fibers" (a vector)
  maps: int or vector: if int, number of activation maps desired; if vector, ids of init sites
  ppm: int of the number of sample points per map
  noise: A factor in milliseconds by which a standard normal distribution of noise 
  is applied to the activation maps
  x0: initial sites 
  """

  def __init__(self, vtk_file, maps=1, ppm=100, noise=0.):
    vf = pv.UnstructuredGrid(vtk_file)
    self.points = vf.points
    self.triangs = vf.cells_dict[vtk.VTK_TRIANGLE]
    self.l = vf.cell_data["fibers"]

    self.n = calculateSurfaceNormalsManifold(self.points, self.triangs)
    self.l = self.l - self.n * np.sum(self.l * self.n, axis=-1, keepdims=True)
    self.l /= np.linalg.norm(self.l, axis=-1, keepdims=True)
    self.t = np.cross(self.l, self.n, axis=-1)
    self.t /= np.linalg.norm(self.t, axis=-1, keepdims=True)
    D_init = (1. ** 2 * self.l[..., np.newaxis] * self.l[..., np.newaxis, :]
        + 1. ** 2 * self.t[..., np.newaxis] * self.t[..., np.newaxis, :]
        + 1. ** 2 * self.n[..., np.newaxis] * self.n[..., np.newaxis, :])
    D_init = 0.5*(D_init + np.transpose(D_init, axes=(0, 2, 1)))

    fim = FIMPY.create_fim_solver(self.points, self.triangs, D_init, device='cpu', use_active_list=False)
    if not np.isscalar(maps):
        x0 = maps
        maps = len(maps)
    else:
        first_point = np.random.choice(self.points.shape[0])
        x0 = [first_point]
        for i in range(maps):
            dist = fim.comp_fim(x0, [0.0]*(i + 1))
            x0.append(np.argmax(dist))

    x0_vals = np.zeros(maps)
    D_n = (.6 ** 2 * self.l[..., np.newaxis] * self.l[..., np.newaxis, :]
        + .4 ** 2 * self.t[..., np.newaxis] * self.t[..., np.newaxis, :]
        + 1e-2 * self.n[..., np.newaxis] * self.n[..., np.newaxis, :])
    D_n = 0.5*(D_n + np.transpose(D_n, axes=(0, 2, 1)))
    self.evecs = np.linalg.eigh(D_n)[1]

    phis = []
    mm = []
    x_e = []
    t_e = []
    inds = [0]
    m_ind = np.random.choice(self.points.shape[0],[ppm,maps],replace=False)
    for i in range(maps):
      phi = fim.comp_fim(x0[i], x0_vals[i], D_n)
      phi = phi + noise * np.random.randn(phi.shape[0])
      m_mask = np.zeros(self.points.shape[0], dtype=bool)
      m_mask[m_ind[:,i]] = True
      phis.append(phi)
      mm.append(m_mask)
      x_e.append(self.points[m_mask])
      t_e.append(phi[m_mask][...,np.newaxis])
      inds.append(len(phi[m_mask]) + inds[-1])
    self.x_e = np.squeeze(np.vstack(x_e))
    self.mm = np.stack(mm, axis=-1)
    self.t_e = np.vstack(t_e)
    self.phis = np.stack(phis, axis=-1)
    self.inds = np.stack(inds)
  
  def get_values(self):
    return self.phis, self.t_e, self.x_e, self.mm, self.evecs, self.inds
