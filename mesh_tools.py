#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

"""
Collection of tools for working with meshes with the vtk library

Functions:  createLocalManifoldBasis,  
            calculateSurfaceNormalsManifold, 
            pointToCellData, cellToPointData
Author: Thomas Grandits
Edited by: Carlos Ruiz
"""

import numpy as np
import pyvista as pv
import vtk
from functools import reduce
import operator

def prod(x):
  """Computes the product over all elements \Prod_{i=1}^n x_i
  """
  return reduce(operator.mul, x, 1)


def createLocalManifoldBasis(points, tris, base_vecs):
  """
  Create a local basis on a triangle basis for manifolds
  The local basis is given the given base_vecs, the triangle
  normal and their orthogonal basis
  """

  assert points.shape[-1] == 3, "Only 3D supported"
  assert tris.shape[-1] == 3, "Only triangles are supported"
  assert base_vecs.shape[0] == tris.shape[0]
  assert np.allclose(np.linalg.norm(base_vecs, axis=-1), 1.), "Basis vectors need to be normalized"

  points_elems = points[tris]
  P = np.empty_like(points_elems)
  surf_normals = calculateSurfaceNormalsManifold(points, tris)
  assert np.allclose(np.sum(base_vecs * surf_normals, axis=-1), 0., atol=2e-5), "Basis vectors and surface normals are not orthonormal"

  orth_vec = np.cross(surf_normals, base_vecs)
  orth_vec /= np.linalg.norm(orth_vec, axis=-1, keepdims=True)
  assert np.allclose(np.sum(orth_vec * surf_normals, axis=-1), 0., atol=1e-6), "Failed to create an orthonormal basis"
  assert np.allclose(np.sum(orth_vec * base_vecs, axis=-1), 0., atol=1e-6), "Failed to create an orthonormal basis"

  P[..., :, 0] = base_vecs
  P[..., :, 1] = orth_vec
  P[..., :, 2] = surf_normals

  return P

def calculateSurfaceNormalsManifold(points, triangs):
  assert triangs.shape[-1] == 3, "Only triangles are supported"
  mesh = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: triangs}, points)
  mesh_surf = mesh.extract_surface().compute_normals()
  assert mesh.n_points == mesh_surf.n_points, "Non manifold vertices detected"
  return mesh_surf.cell_data["Normals"][mesh_surf.cell_data["vtkOriginalCellIds"]]

def pointToCellData(points, elems, point_data):
  assert elems.shape[-1] == 3, "Only triangles are supported"
  mesh = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: elems}, points)
  orig_shape = point_data.shape[1:]
  mesh.point_data["data"] = point_data.reshape([-1, prod(orig_shape)])
  return mesh.point_data_to_cell_data().cell_data["data"].reshape((-1,) + orig_shape)

def cellToPointData(points, elems, cell_data):
  assert elems.shape[-1] == 3, "Only triangles are supported"
  mesh = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: elems}, points)
  orig_shape = cell_data.shape[1:]
  mesh.cell_data["data"] = cell_data.reshape([-1, prod(orig_shape)])
  return mesh.cell_data_to_point_data().point_data["data"].reshape((-1,) + orig_shape)
