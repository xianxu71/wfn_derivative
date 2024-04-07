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
wfn_file_yp = h5.File('wfn_yp.h5','r')
wfn_file_yn = h5.File('wfn_yn.h5','r')
wfn_file_zp = h5.File('wfn_zp.h5','r')
wfn_file_zn = h5.File('wfn_zn.h5','r')


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

gvecs_xn = wfn_file_xn['wfns/gvecs'][()]
coeffs_xn = wfn_file_xn['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file_xn['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk_xn = wfn_file_xn['mf_header/kpoints/ngk'][()]
k_index_xn =np.hstack((np.array([0]), np.cumsum(ngk_xn)))

gvecs_yp = wfn_file_yp['wfns/gvecs'][()]
coeffs_yp = wfn_file_yp['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file_yp['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk_yp = wfn_file_yp['mf_header/kpoints/ngk'][()]
k_index_yp =np.hstack((np.array([0]), np.cumsum(ngk_yp)))

gvecs_yn = wfn_file_yn['wfns/gvecs'][()]
coeffs_yn = wfn_file_yn['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file_yn['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk_yn = wfn_file_yn['mf_header/kpoints/ngk'][()]
k_index_yn =np.hstack((np.array([0]), np.cumsum(ngk_yn)))

gvecs_zp = wfn_file_zp['wfns/gvecs'][()]
coeffs_zp = wfn_file_zp['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file_zp['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk_zp = wfn_file_zp['mf_header/kpoints/ngk'][()]
k_index_zp =np.hstack((np.array([0]), np.cumsum(ngk_zp)))

gvecs_zn = wfn_file_zn['wfns/gvecs'][()]
coeffs_zn = wfn_file_zn['wfns/coeffs'][novb-nv:novb+nc, :, :, 0]+1j*wfn_file_zn['wfns/coeffs'][novb-nv:novb+nc, :, :, 1]
ngk_zn = wfn_file_zn['mf_header/kpoints/ngk'][()]
k_index_zn =np.hstack((np.array([0]), np.cumsum(ngk_zn)))

for ik in range(nk):
    gvecs_k = gvecs[k_index[ik]:k_index[ik + 1], :]
    coeffs_k = coeffs[:, :, k_index[ik]:k_index[ik + 1]]

    gvecs_k_xp = gvecs_xp[k_index_xp[ik]:k_index_xp[ik + 1], :]
    coeffs_k_xp = coeffs_xp[:, :, k_index_xp[ik]:k_index_xp[ik + 1]]

    gvecs_k_xn = gvecs_xn[k_index_xn[ik]:k_index_xn[ik + 1], :]
    coeffs_k_xn = coeffs_xn[:, :, k_index_xn[ik]:k_index_xn[ik + 1]]

    gvecs_k_yp = gvecs_yp[k_index_yp[ik]:k_index_yp[ik + 1], :]
    coeffs_k_yp = coeffs_yp[:, :, k_index_yp[ik]:k_index_yp[ik + 1]]

    gvecs_k_yn = gvecs_yn[k_index_yn[ik]:k_index_yn[ik + 1], :]
    coeffs_k_yn = coeffs_yn[:, :, k_index_yn[ik]:k_index_yn[ik + 1]]

    gvecs_k_zp = gvecs_zp[k_index_zp[ik]:k_index_zp[ik + 1], :]
    coeffs_k_zp = coeffs_zp[:, :, k_index_zp[ik]:k_index_zp[ik + 1]]

    gvecs_k_zn = gvecs_zn[k_index_zn[ik]:k_index_zn[ik + 1], :]
    coeffs_k_zn = coeffs_zn[:, :, k_index_zn[ik]:k_index_zn[ik + 1]]



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

    gspace_k_dic_yp = create_hash_table(gvecs_k_yp)
    new_index_k_yp = get_index(gspace_k_dic_yp, gvecs_k)

    coeffs_k_yp_new = coeffs_k_yp[:, :, new_index_k_yp]
    yesorno_yp = np.where(new_index_k_yp == -1)
    coeffs_k_yp_new[:, :, yesorno_yp] = 0

    gspace_k_dic_yn = create_hash_table(gvecs_k_yn)
    new_index_k_yn = get_index(gspace_k_dic_yn, gvecs_k)

    coeffs_k_yn_new = coeffs_k_yn[:, :, new_index_k_yn]
    yesorno_yn = np.where(new_index_k_yn == -1)
    coeffs_k_yn_new[:, :, yesorno_yn] = 0

    gspace_k_dic_zp = create_hash_table(gvecs_k_zp)
    new_index_k_zp = get_index(gspace_k_dic_zp, gvecs_k)

    coeffs_k_zp_new = coeffs_k_zp[:,:,new_index_k_zp]
    yesorno_zp = np.where(new_index_k_zp == -1)
    coeffs_k_zp_new[:,:,yesorno_zp] = 0

    gspace_k_dic_zn = create_hash_table(gvecs_k_zn)
    new_index_k_zn = get_index(gspace_k_dic_zn, gvecs_k)

    coeffs_k_zn_new = coeffs_k_zn[:, :, new_index_k_zn]
    yesorno_zn = np.where(new_index_k_zn == -1)
    coeffs_k_zn_new[:, :, yesorno_zn] = 0

    overlap_xp = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_xp_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_xp_new[:,1,:],optimize='optimal')

    overlap_xn = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_xn_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_xn_new[:,1,:],optimize='optimal')

    overlap_yp = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_yp_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_yp_new[:,1,:],optimize='optimal')

    overlap_yn = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_yn_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_yn_new[:,1,:],optimize='optimal')

    overlap_zp = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_zp_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_zp_new[:,1,:],optimize='optimal')

    overlap_zn = np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,0,:]),coeffs_k_zn_new[:,0,:])  \
                +np.einsum("mg,ng -> mn",np.conj(coeffs_k[:,1,:]),coeffs_k_zn_new[:,1,:],optimize='optimal')

    connected_overlap_xp = overlap_xp*(abs(overlap_xp)>0.1)
    connected_overlap_xn = overlap_xn*(abs(overlap_xn)>0.1)
    connected_overlap_yp = overlap_yp*(abs(overlap_yp)>0.1)
    connected_overlap_yn = overlap_yn*(abs(overlap_yn)>0.1)
    connected_overlap_zp = overlap_zp*(abs(overlap_zp)>0.1)
    connected_overlap_zn = overlap_zn*(abs(overlap_zn)>0.1)

    Uxp, Sxp, Vhxp = np.linalg.svd(connected_overlap_xp)
    Uxn, Sxn, Vhxn = np.linalg.svd(connected_overlap_xn)

    Uyp, Syp, Vhyp = np.linalg.svd(connected_overlap_yp)
    Uyn, Syn, Vhyn = np.linalg.svd(connected_overlap_yn)

    Uzp, Szp, Vhzp = np.linalg.svd(connected_overlap_zp)
    Uzn, Szn, Vhzn = np.linalg.svd(connected_overlap_zn)

    corrected_overlap_xp = overlap_xp@(Vhxp.T.conj()@Uxp.T.conj())
    corrected_overlap_xn = overlap_xn@(Vhxn.T.conj()@Uxn.T.conj())

    corrected_overlap_yp = overlap_yp@(Vhyp.T.conj()@Uyp.T.conj())
    corrected_overlap_yn = overlap_yn@(Vhyn.T.conj()@Uyn.T.conj())

    corrected_overlap_zp = overlap_zp@(Vhzp.T.conj()@Uzp.T.conj())
    corrected_overlap_zn = overlap_zn@(Vhzn.T.conj()@Uzn.T.conj())

    dipole_matrix[ik,:,:,0] = 1j*(corrected_overlap_xp-corrected_overlap_xn)/np.sqrt(bdot[0,0])/q/2#*2 shouldn't *2 if you want p instead of v
    dipole_matrix[ik,:,:,1] = 1j*(corrected_overlap_yp-corrected_overlap_yn)/np.sqrt(bdot[1,1])/q/2#*2 shouldn't *2 if you want p instead of v
    dipole_matrix[ik,:,:,2] = 1j*(corrected_overlap_zp-corrected_overlap_zn)/np.sqrt(bdot[2,2])/q/2#*2 shouldn't *2 if you want p instead of v
    print(str(ik)+"/"+str(nk))


h5_file_w = h5.File('dipole_matrix.h5','w')
h5_file_w.create_dataset('dipole', data=dipole_matrix)
h5_file_w.close()




