#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

"""
Collection of tools to optimize and group mathematical calculations

Functions:  matMulProdSum, 
            eigenDecompProd,
            metricNormMatrix
Author: Thomas Grandits
Edited by: Carlos Ruiz
"""

import tensorflow as tf

def matMulProdSum(A, B):
  """Computes A @ B in a broadcasted fashion
  """
  return tf.einsum('...xy,...yz->...xz', A, B)

def eigenDecompProd(A, B):
  """Computes the eigenreconstruction of a tensor A * B * A^T
  """

  result = matMulProdSum(matMulProdSum(A, B), tf.transpose(A, perm=[0, 2, 1]))

  return result

def metricNormMatrix(A, x1, x2=None, ret_sqrt=True):
  """Computes \sqrt{<<x1.T, A>, x2>} with or without the sqrt. If x2 is not set, x2 = x1
  """
  if x2 is None:
    x2 = x1

  sqr_norm = tf.reduce_sum(tf.reduce_sum(A * x1[..., tf.newaxis], axis=-2) * x2, axis=-1)

  return (tf.sqrt(sqr_norm) if ret_sqrt else sqr_norm)
