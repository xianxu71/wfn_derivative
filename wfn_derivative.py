import numpy as np
import h5py as h5
import time

def create_hash_table(arrays):
    hash_table = {}
    for index, array in enumerate(arrays):
        hash_table[tuple(array)] = int(index)
    return hash_table
def get_index(hash_table, arrays):
    new_index = np.zeros(arrays.shape[0],dtype=int)
    for index, array in enumerate(arrays):
        new_index[index] = (hash_table.get(tuple(array),-1))
    return new_index


novb = 46
nv = 8
nc = 8
q = 0.001

wfn_file = h5.File('wfn_0.h5','r')
wfn_file_xp = h5.File('wfn_xp.h5','r')
wfn_file_xn = h5.File('wfn_xn.h5','r')

a =wfn_file['/mf_header/crystal/alat'][()]*wfn_file['/mf_header/crystal/avec'][()]*0.52917
bdot =wfn_file['/mf_header/crystal/bdot'][()]
nk = wfn_file['mf_header/kpoints/nrk'][()]
nb = nv+nc
energy = wfn_file['mf_header/kpoints/el'][0,:,novb-nv:novb+nc]
rk = wfn_file['mf_header/kpoints/rk'][()]
dipole_matrix = np.zeros([nk,nb,nb,3],dtype=np.complex128)

#0
gvecs = wfn_file['wfns/gvecs'][()]
coeffs = wfn_file['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk = wfn_file['mf_header/kpoints/ngk'][()]
k_index =np.hstack((np.array([0]), np.cumsum(ngk)))
ng = wfn_file['/mf_header/gspace/ng'][()]

#xp
gvecs_xp = wfn_file_xp['wfns/gvecs'][()]
coeffs_xp = wfn_file_xp['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file_xp['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk_xp = wfn_file_xp['mf_header/kpoints/ngk'][()]
k_index_xp =np.hstack((np.array([0]), np.cumsum(ngk_xp)))
#ng_xp = wfn_file_xp['/mf_header/gspace/ng'][()]

gvecs_xn = wfn_file_xn['wfns/gvecs'][()]
coeffs_xn = wfn_file_xn['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file_xn['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk_xn = wfn_file_xn['mf_header/kpoints/ngk'][()]
k_index_xn =np.hstack((np.array([0]), np.cumsum(ngk_xn)))

for ik in range(nk):
    gvecs_k = gvecs[k_index[ik]:k_index[ik + 1], :]
    coeffs_k = coeffs[:, :, k_index[ik]:k_index[ik + 1]]

    gvecs_k_xp = gvecs_xp[k_index_xp[ik]:k_index_xp[ik + 1], :]
    coeffs_k_xp = coeffs_xp[:, :, k_index_xp[ik]:k_index_xp[ik + 1]]

    gvecs_k_xn = gvecs_xn[k_index_xn[ik]:k_index_xn[ik + 1], :]
    coeffs_k_xn = coeffs_xn[:, :, k_index_xn[ik]:k_index_xn[ik + 1]]


    gspace_k_dic_xp = create_hash_table(gvecs_k_xp)
    new_index_k_xp = get_index(gspace_k_dic_xp, gvecs_k)

    coeffs_k_xp_new = coeffs_k_xp[:,:,new_index_k_xp]
    yesorno_xp = np.where(new_index_k_xp == -1)
    coeffs_k_xp_new[:,:,yesorno_xp] = 0


    gspace_k_dic_xn = create_hash_table(gvecs_k_xn)
    new_index_k_xn = get_index(gspace_k_dic_xn, gvecs_k)

    coeffs_k_xn_new = coeffs_k_xn[:,:,new_index_k_xn]
    yesorno_xn = np.where(new_index_k_xn == -1)
    coeffs_k_xn_new[:,:,yesorno_xn] = 0

    overlap_xp = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_xp_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_xp_new[:,1,:],optimize='optimal')

    overlap_xn = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_xn_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_xn_new[:,1,:],optimize='optimal')

    connected_overlap_xp = overlap_xp*(abs(overlap_xp)>0.1)
    connected_overlap_xn = overlap_xn*(abs(overlap_xn) > 0.1)

    Uxp, Sxp, Vhxp = np.linalg.svd(connected_overlap_xp)

    Uxn, Sxn, Vhxn = np.linalg.svd(connected_overlap_xn)

    corrected_overlap_xp = overlap_xp@(Vhxp.T.conj()@Uxp.T.conj())
    corrected_overlap_xn = overlap_xn@(Vhxn.T.conj()@Uxn.T.conj())

    dipole_matrix[ik,:,:,0] = 1j*(corrected_overlap_xp-corrected_overlap_xn)/np.sqrt(bdot[0,0])/q/2#*2 shouldn't *2 if you want p instead of v
    print(str(ik)+"/"+str(nk))


h5_file_w = h5.File('dipole_matrix.h5','w')
h5_file_w.create_dataset('dipole', data=dipole_matrix)
h5_file_w.close()




