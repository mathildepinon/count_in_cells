import os
import copy

import numpy as np
import pandas
import matplotlib as mpl
from matplotlib import pyplot as plt

from pypower import CatalogMesh
from pypower.utils import sky_to_cartesian
from mockfactory import Catalog, RedshiftDensityInterpolator, utils
from cosmoprimo import fiducial


# Count-in-cells in cubic box
def compute_cic_density(data, smoothing_radius, cellsize=None, shift=False, use_rsd=False, use_weights=False):
    boxsize = data.boxsize
    offset = data.boxcenter - data.boxsize/2.
    
    if use_rsd and data.positions_rsd is not None:
        positions = data.positions_rsd
    else:
        positions = data.positions
        
    if use_weights and data.weights is not None:
        weights = data.weights
        norm = np.sum(weights) * (4/3 * np.pi * smoothing_radius**3) / boxsize**3
    else:
        weights = None
        norm = data.size * (4/3 * np.pi * smoothing_radius**3) / boxsize**3

    if cellsize is None:
        cellsize = smoothing_radius * 2
    else:
        if cellsize < 2 * smoothing_radius:
            print("Cellsize must be bigger than twice the smoothing radius.")
    
    if shift:
        indices_in_grid = ((positions - offset) / cellsize).astype('i4')
        grid_pos = (indices_in_grid + 0.5) * cellsize + offset
        dist_to_nearest_node = np.sum((grid_pos - positions)**2, axis=0)**0.5
        mask_particles = dist_to_nearest_node < smoothing_radius

        nmesh = np.int32(boxsize / cellsize)
        mask_particles &= np.all((indices_in_grid >= 0) & (indices_in_grid < nmesh), axis=0)
        mesh = np.zeros((nmesh,)*3, dtype='f8')
        np.add.at(mesh, tuple(indices_in_grid[:, mask_particles]), weights[mask_particles] if use_weights else 1.)
        return mesh / norm - 1
        
    else:
        def compute_density_mesh(pos):
            indices_in_grid = ((pos - offset) / cellsize + 0.5).astype('i4')
            grid_pos = indices_in_grid * cellsize + offset
            dist_to_nearest_node = np.sum((grid_pos - positions)**2, axis=0)**0.5
            mask_particles = dist_to_nearest_node < smoothing_radius

            nmesh = np.int32(boxsize / cellsize)
            mask_particles &= np.all((indices_in_grid > 0) & (indices_in_grid < nmesh), axis=0)
            mesh = np.zeros((nmesh - 1,)*3, dtype='f8')
            np.add.at(mesh, tuple(indices_in_grid[:, mask_particles] - 1), weights[mask_particles] if use_weights else 1.)
            return mesh
        
        data_mesh = compute_density_mesh(positions)
        mesh = data_mesh / norm - 1
            
        return mesh
    

def get_rdd(catalog, cosmo=fiducial.DESI()):
    ra, dec, z = catalog['RA'], catalog['DEC'], catalog['Z']
    return [ra, dec, cosmo.comoving_radial_distance(z)]


# Count-in-cells for realistic survey
def compute_cic_density_survey(tracer, completeness, region, smoothing_radius, cellsize=None, th=0, nmocks=1, use_rsd=False):
    
    if cellsize is None:
        cellsize = smoothing_radius * 2
    else:
        if cellsize < 2 * smoothing_radius:
            print("Cellsize must be bigger than twice the smoothing radius.")
            
    zlim = {'ELG': (0.8, 1.6), 'LRG':(0.4, 1.1), 'QSO':(0.8, 3.5)}
    
    mesh_list = list()
    
    for imock in range(nmocks):
        data_fn = "/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats/{}_{}{}_clustering.dat.fits".format(imock, tracer, completeness, region)
        randoms_fn = "/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1v1/mock{:d}/LSScats/{}_{}{}_0_clustering.ran.fits".format(imock, tracer, completeness, region)
        data = Catalog.read(data_fn, filetype='fits')
        randoms = Catalog.read(randoms_fn, filetype='fits')
        
        # redshift cut
        data = data[(data['Z'] > zlim[tracer][0]) & (data['Z'] < zlim[tracer][1])]
        randoms = randoms[(randoms['Z'] > zlim[tracer][0]) & (randoms['Z'] < zlim[tracer][1])]
        
        # convert coordinates to cartesian
        data_positions = sky_to_cartesian(get_rdd(data))
        randoms_positions = sky_to_cartesian(get_rdd(randoms))
        data_weights = data['WEIGHT']
        randoms_weights = randoms['WEIGHT']

        offset = np.min(data_positions, axis=1)
        boxsize = np.max(data_positions, axis=1) - offset
            
        def compute_density_mesh(pos, weights=None):
            pos = pos - offset[:, None]
            indices_in_grid = np.int32(pos / cellsize)
            grid_pos = indices_in_grid * cellsize
            dist_to_nearest_node = np.sum((grid_pos - pos)**2, axis=0)**0.5
            mask_particles = dist_to_nearest_node < smoothing_radius

            nmesh = np.int32(boxsize / cellsize)
            mesh = np.zeros(nmesh, dtype='f8')
            mask_particles &= np.all(indices_in_grid < nmesh[:, None], axis=0)
            np.add.at(mesh, tuple(indices_in_grid[:, mask_particles]), weights[mask_particles] if (weights is not None) else 1.)
            return mesh

        data_mesh = compute_density_mesh(data_positions, data_weights)
        randoms_mesh = compute_density_mesh(randoms_positions, randoms_weights)
        mesh = np.zeros_like(data_mesh)
        non_zeros_indices = np.logical_and(data_mesh != 0, randoms_mesh > th)
        nan_indices = np.where(randoms_mesh <= th)
        mesh[non_zeros_indices] = data_mesh[non_zeros_indices] * randoms_weights.csum() / (randoms_mesh[non_zeros_indices] * data_weights.csum())
        mesh[nan_indices] = np.nan
        mesh_list.append(mesh - 1)
    #np.save('test', np.array(mesh_list))
    
    return np.concatenate([mesh_list[i].flatten() for i in range(nmocks)])


if __name__ == '__main__':
    from cosmoprimo import fiducial
    cosmo = fiducial.DESI()

    nmocks = 25
    tracer = "ELG"
    completeness = 'complete_'
    region = 'S'
    z_avg = {'ELG': 1.1, 'LRG': 0.8, 'QSO': 1.4}
    z = z_avg[tracer]
    
    # Density smoothing parameters
    cellsize = 30
    R = cellsize / 2
    
    th = 5
    density_cic = compute_cic_density_survey(tracer, completeness, region=region, smoothing_radius=R, cellsize=cellsize, th=th, nmocks=nmocks, use_rsd=False)
    
    np.save('density_cic_{}mocks_{}_{}{}_R{}Mpc_{}th'.format(nmocks, tracer, completeness, region, R, th), density_cic)
    