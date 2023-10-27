# Class for Exact Diagonalization Calculations
import numpy as np
from scipy import sparse
import functools as ft
import itertools as it
from numba import jit

# tensor product of list of operators (sparse)
def tp(mat_list):
    return ft.reduce(sparse.kron, mat_list)

class SpinChain():
    def __init__(self, spin, L, bc=0, flux=0):
        self.spin = spin # spin can be any half-integer
        self.q = int(2*spin + 1) # local Hilbert space dim
        self.L = L # system size (length)
        self.Ham = sparse.bsr_array((self.q**L, self.q**L), dtype=complex) # hamiltonian
        s0 = np.eye(self.q) 
        sz = np.diag(np.arange(-spin, spin+1, 1.0)) 
        sp = np.diag([np.sqrt(spin*(spin+1) - m*(m+1)) 
                    for m in np.arange(-spin, spin, 1.0)], k=1) 
        sm = np.diag([np.sqrt(spin*(spin+1) - m*(m+1)) 
                    for m in np.arange(-spin, spin, 1.0)], k=-1) 
        self.spin_mat_list = [s0, sz, sp, sm] # spin matrices

        # Possible boundary conditions (bc) are 0 (open), 1 (periodic), or -1 (antiperiodic)
        # flux is an additional twist (multiplicative phase) applied to the boundary condtions
        self.bc = bc * np.exp(1.j*flux)

        # keep track of projector if we project to a sector of the Hilbert space
        self.projector = None
        # keep track of full hamiltonian if we project to a sector of the Hilbert space
        self.FullHam = None

    # returns identity and spin matrices 
    # note that spin matrices by default include factor of spin (1/2 for spin-1/2, etc.)
    def getPauli(self, basis="pm"):
        # return identity, sz, raising operator sp, lowering operator sm
        if basis == "pm":
            return self.spin_mat_list
        # return identity, sx, sy, sz (spin matrices)
        elif basis == "xyz":
            s0, sz, sp, sm = self.spin_mat_list
            sx = 1/2 * (sp + sm)
            sy = 1/(2j) * (sp - sm)
            return [s0, sx, sy, sz]
        # return identity, X, Y, Z Pauli matrices (2 x 2)
        elif basis == 'pauli':
            sigmax = np.array([[0, 1], [1, 0]])
            sigmay = np.array([[0, -1j], [1j, 0]])
            sigmaz = np.array([[1, 0], [0, -1]])
            return [np.eye(self.q), sigmax, sigmay, sigmaz]
        # return identity (I) and clock (Z) and shift (X) operators 
        elif basis == 'clock_shift':
            w = np.exp(2*np.pi*1j/self.q)
            Z = np.diag([w**n for n in range(self.q)])
            X = np.roll(np.eye(self.q), -1, axis=0)
            return [np.eye(self.q), Z, X]

    # returns Hamiltonian (which is a sparse array)
    def getHam(self):
        return self.Ham

    # return dimensions of Hilbert space (or subspace if Hamiltonian is projected)
    def getHilbertDim(self):
        return self.Ham.shape[0]

    # local spin list is associated with each basis element 
    # (e.g., element 0 -> |00000>, element 1 -> |00001>, element 2 -> |00011>)
    # returns array of spin lists
    # row index = basis element number, col index = site number, entry = spin
    def basis_to_occ(self):
        # much faster if we can use binary operations
        q, L = self.q, self.L
        if q==2:
            # make column vector of basis element indices
            basis_arr = np.arange(2**L).reshape([-1,1])
            # make row vector of powers of two (increasing R to L)
            mask = 2**np.arange(L)[::-1]
            # apply bitwise "and" between elements of basis_arr and mask
            # (e.g. 5 & 4 -> 101 & 100 = 100 -> 4)
            # each row is now like [5&8, 5&4, 5&2, 5&1] = [0, 4, 0, 1]
            # converting nonzero entries to 1 gives list of binary digits of 5 -> [0,1,0,0] 
            occ_list = (basis_arr & mask).astype(bool).astype("int8")
        # slightly slower for spin-1 and beyond, use numba to speed up
        else:
            @jit(nopython=True, cache=True)
            def return_answer():
                # row vector of powers of q (decreasing)
                powers_of_q = q**np.arange(L)[::-1]
                # column vector of basis state indices
                basis_arr = np.arange(q**L).reshape((-1,1))
                # answer is (q**L, L) array
                # row of index A is the digits of A base-q
                return (basis_arr // powers_of_q % q)
            occ_list = return_answer()
        # subtract total spin to set range as (-spin, spin) instead of (0, 2*spin) 
        return occ_list-self.spin

    # returns magnetization and or dipole of all basis elements
    def mag_dip_basis_list(self, mag=True, dip=True):
        bto = self.basis_to_occ()
        if mag==True and dip==False:
            mag_basis = np.sum(bto, axis=1)
            return mag_basis
        elif mag==False and dip==True:
            dip_basis = np.einsum('ij,j->i', bto, np.arange(self.L))
            return dip_basis
        elif mag==True and dip==True:
            mag_basis = np.sum(bto, axis=1)
            dip_basis = np.einsum('ij,j->i', bto, np.arange(self.L))
            return (mag_basis, dip_basis)
        else:
            print("SpinChain Error: either mag or dip must be True")   
    
    # operators (numpy arrays) in op_list applied to sites (integers) at site_list
    def op_at_sites(self, op_list, site_list, coeff=1):
        # list of operator labels at each site
        # identity is labeled by string "id", specified operators labeled by their index in op_list
        # e.g. ["id", "id", "0", "1", "id"] or ["2", "id", "id", "0", "1"] for PBC
        site_list = np.array(site_list) % self.L # make site list mod_L
        label_list = ["id" for site in range(self.L)]
        for op_label, site in enumerate(site_list):
            label_list[site] = op_label
        # group contiguous identity operators together
        label_list = [list(group) for key, group in it.groupby(label_list)]
        # make list of matrices to tensor together
        mat_list = []
        for group in label_list:
            if "id" in group:
                occurences = len(group)
                mat_list.append(sparse.eye(self.q**occurences)) # group of local identities
            else:
                op_label = group[0]
                mat_list.append(op_list[op_label]) # specified operator in op_list, site_list
        # take tensor product of resulting matrices
        return coeff*tp(mat_list)

    # return global magnetization operator
    def getMagOp(self):
        sz = self.spin_mat_list[1]
        Sz = self.op_at_sites([sz], [0])
        for i in range(1, self.L):
            Sz += self.op_at_sites([sz],[i])
        return Sz
    
    # return global dipole operator 
    def getDipOp(self):
        sz = self.spin_mat_list[1]
        Pz = self.op_at_sites([0*sz], [0])
        for i in range(1, self.L):
            Pz += i * self.op_at_sites([sz],[i])
        return Pz

    # e.g. op_list = [sp, sm] and site_list_list = [[0,1], [1,2], [2,3], ...]
    # make sure the site_lists are in ascending order if within chain
    def setHamTerm(self, op_list, site_list_list, coeff=1, ignore_bc=False):
        for site_list in site_list_list:
            site_list = np.array(site_list) % self.L          
            # site in site_list that first wraps around chain
            wrap_index = np.where(np.diff(site_list)<0)[0]
            # keep track of boundary conditions for sites in couplings that wrap around chain
            if (wrap_index.size > 0) and (ignore_bc == False):
                # total number of sites that wrap around the chain 
                num_wrap = len(site_list[wrap_index[0]:])-1 
                self.Ham = self.Ham + self.bc**num_wrap * self.op_at_sites(op_list, site_list, coeff=coeff)
            else:
                self.Ham = self.Ham + self.op_at_sites(op_list, site_list, coeff=coeff)

    # does e^g ham + e^-g ham^\dag with nonHermiticity set by g
    def plusConj(self, g=0):
        self.Ham = np.exp(g)*self.Ham +  np.exp(-g)*self.Ham.conj().T

    # for spin-1/2 returns magnetization of n-particle sector
    def particleToMag(self, n):
        return (n - self.L/2)

    # projector onto magnetization sector
    # not square matrix H -> P H (P.T)
    def P_mag(self, M):
        # list of magnetizations of basis states
        mag_list = self.mag_dip_basis_list(mag=True, dip=False)
        # indices of basis states in desired sector
        indices = np.where(np.abs(mag_list - M) < 1e-9)[0]
        P = sparse.lil_array((len(indices), self.q**self.L))
        for i in range(len(indices)):
            P[i, indices[i]] = 1.0
        return P

    # projector onto dipole sector
    # include option to define dipole moment periodically mod L (valid for PBC)
    def P_dip(self, Pdip, periodic_dip=True):
        # list of dipoles of basis states
        dip_list = self.mag_dip_basis_list(mag=False, dip=True)
        # indices of basis states in desired sector
        # define dipole moment mod L for periodic boundary conditions
        if periodic_dip == True:
            indices = np.where(np.abs((dip_list - Pdip)%self.L) < 1e-9)[0]
        else:
            indices = np.where(np.abs(dip_list - Pdip) < 1e-9)[0]        

        P = sparse.lil_array((len(indices), self.q**self.L))
        for i in range(len(indices)):
            P[i, indices[i]] = 1.0
        return P

    # projector onto spin and dipole sector
    # include option to define dipole moment periodically mod L (valid for PBC)
    def P_mag_dip(self, M, Pdip, periodic_dip=False):
        # list of magnetizations and dipole moments of basis elements
        mag_list, dip_list = self.mag_dip_basis_list(mag=True, dip=True)
        # indices of basis states in desired sector
        # project to Pdip (mod L) if we have PBC because shifting dipole is only well-defined mod L (and with M=0)
        if periodic_dip == True:
            indices = np.where(np.abs(mag_list - M) + np.abs((dip_list - Pdip)%self.L) < 1e-9)[0]
        else:
            indices = np.where(np.abs(mag_list - M) + np.abs(dip_list - Pdip) < 1e-9)[0]

        P = sparse.lil_array((len(indices), self.q**self.L))
        for i in range(len(indices)):
            P[i, indices[i]] = 1.0
        return P

    # project onto specific subspace labeled by state indices
    def P_specific(self, state_index_list):
        P = sparse.lil_array((len(state_index_list), self.q**self.L))
        for i in range(len(state_index_list)):
            P[i, int(state_index_list[i])] = 1.0
        return P

    # project Hamiltonian a magnetization and/or dipole sector
    def projectHam(self, M='none', Pdip='none', periodic_dip=False):
        if M==Pdip=='none':
            P = sparse.csr_array(np.eye(self.q**self.L))
        elif M!='none' and Pdip=='none':
            P = self.P_mag(M).tocsr()
        elif M=='none' and Pdip!='none':
            P = self.P_dip(Pdip).tocsr()
        else:
            P = self.P_mag_dip(M, Pdip, periodic_dip).tocsr()
        self.FullHam = self.Ham.copy()
        self.Ham = P @ self.Ham @ P.T
        self.projector = P

    # project Hamiltonian into specified set of basis states
    def projectHamSpecific(self, state_index_list):
        P = self.P_specific(state_index_list).tocsr()
        self.FullHam = self.Ham.copy()
        self.Ham = P @ self.Ham @ P.T
        self.projector = P

    # unproject the hamiltonian and bring it back to full hilbert space
    def unprojectHam(self):
        self.Ham = self.FullHam.copy()
        self.projector = None
        self.FullHam = None

    def getProjector(self):
        return self.projector

    # entanglement entropy w/ region A = [0, x] and region B = [x, L]
    # works for single state or multiple states (inputted as columns of an array)
    def getEntEntropy(self, state, subsyst_size):
        # put state into full Hilbert space
        if len(state.shape) == 1:
            num_states = 1
        else:
            num_states = state.T.shape[0]
        if self.projector != None:
            state = self.projector.T @ state
        # reshape states from columns of matrix into rows if many are present
        # reshape into matrix separating |psi> = ∑ w_{ij} |i>_{left half} |j>_{right half}
        # first index is state index
        state_mat = np.reshape(state.T, (num_states, self.q**subsyst_size, -1))
        # find singular values to get Schmidt decomposition
        # if there are many states then svd is computed for last two indices (correct)
        svals = np.linalg.svd(state_mat, compute_uv=False)
        # make all singular values slightly positive to take log
        svals += 1e-20
        vn_entropy = -np.sum(svals**2 * np.log(svals**2), axis=1)
        
        return vn_entropy if num_states > 1 else vn_entropy[0]

    # One dimensional (SPT and SSB) version of topological entanglement entropy
    # Xiao-Gang Wen definition, split chain into A, B, D, C subregions
    # currently only works for a single state !
    # S_AB + S_BC - S_B - S_ABC
    def getStopo(self, state):
        # put state into full Hilbert space before reshaping into tensor product structure
        q, L = self.q, self.L
        A, B, D, C= [*range(0, L//4)], [*range(L//4, L//2)], [*range(L//2, 3*L//4)], [*range(3*L//4, L)]
        if self.projector != None:
            state = self.projector.T.toarray() @ state
        psi = np.reshape(state, L*[q])
        # S_AB
        psi_block = np.reshape(psi, (2**(len(A)+len(B)), -1))
        svals = np.linalg.svd(psi_block, compute_uv=False)
        svals_pos = svals[svals > 1e-20]
        S_AB = -np.sum(svals_pos**2 * np.log(svals_pos**2))
        # S_BC
        psi_block = np.reshape(np.transpose(psi, (B + C) + A + D), (2**(len(B)+len(C)), -1))
        svals = np.linalg.svd(psi_block, compute_uv=False)
        svals_pos = svals[svals > 1e-20]
        S_BC = -np.sum(svals_pos**2 * np.log(svals_pos**2))
        # S_B
        psi_block = np.reshape(np.transpose(psi, (B) + A + D + C), (2**len(B), -1))
        svals = np.linalg.svd(psi_block, compute_uv=False)
        svals_pos = svals[svals > 1e-20]
        S_B = -np.sum(svals_pos**2 * np.log(svals_pos**2))
        # S_ABC
        psi_block = np.reshape(np.transpose(psi, (A + B + C) + D), (2**(len(A)+len(B)+len(C)), -1))
        svals = np.linalg.svd(psi_block, compute_uv=False)
        svals_pos = svals[svals > 1e-20]
        S_ABC = -np.sum(svals_pos**2 * np.log(svals_pos**2))
        # Mutual information between two sites
        return (S_AB + S_BC - S_B - S_ABC)

    # Rényi entanglement entropy w/ region A = [0, x] and region B = [x, L]
    def getRenyi(self, state, subsyst_size, alpha):
        # put state into full Hilbert space
        if len(state.shape) == 1:
            num_states = 1
        else:
            num_states = state.T.shape[0]
        if self.projector != None:
            state = self.projector.T.toarray() @ state
        # reshape states from columns of matrix into rows if many are present
        # reshape into matrix separating |psi> = ∑ w_{ij} |i>_{left half} |j>_{right half}
        # first index is state index
        state_mat = np.reshape(state.T, (num_states, self.q**subsyst_size, -1))
        # find singular values to get Schmidt decomposition
        # if there are many states then svd is computed for last two indices (correct)
        svals = np.linalg.svd(state_mat, compute_uv=False)
        # make all singular values slightly positive to take log
        svals += 1e-20
        renyi_entropy = np.log(np.sum(svals**(2*alpha), axis=1))/(1-alpha)
        
        return renyi_entropy if num_states > 1 else renyi_entropy[0]


    # mutual information between two sites
    # currently only works for a single state !
    def getMutualInfo(self, state, i, j):
        # put state into full Hilbert space before reshaping into tensor product structure
        q, L = self.q, self.L
        if self.projector != None:
            state = self.projector.T.toarray() @ state
        psi = np.reshape(state, L*[q])
        other_index_list = [k for k in range(L) if k not in [i,j]]
        # S_i
        psi_block = np.reshape(np.transpose(psi,[i,j]+other_index_list),(q, q**(L-1)))
        svals = np.linalg.svd(psi_block, compute_uv=False)
        svals_pos = svals[svals > 1e-20]
        Si = -np.sum(svals_pos**2 * np.log(svals_pos**2))
        # S_j
        psi_block = np.reshape(np.transpose(psi,[j,i]+other_index_list),(q, q**(L-1)))
        svals = np.linalg.svd(psi_block, compute_uv=False)
        svals_pos = svals[svals > 1e-20]
        Sj = -np.sum(svals_pos**2 * np.log(svals_pos**2))
        # S_ij
        psi_block = np.reshape(np.transpose(psi,[i,j]+other_index_list),(q**2, q**(L-2)))
        svals = np.linalg.svd(psi_block, compute_uv=False)
        svals_pos = svals[svals > 1e-20]
        Sij = -np.sum(svals_pos**2 * np.log(svals_pos**2))
        # Mutual information between two sites
        return (Si + Sj  - Sij)

    # entanglement spectrum of density matrix
    def getEntSpectrum(self, state, subsyst_size):
        # put state into full Hilbert space
        if len(state.shape) == 1:
            num_states = 1
        else:
            num_states = state.T.shape[0]
        if self.projector != None:
            state = self.projector.T.toarray() @ state
        # reshape states from columns of matrix into rows if many are present
        # reshape into matrix separating |psi> = ∑ w_{ij} |i>_{left half} |j>_{right half}
        # first index is state index
        state_mat = np.reshape(state.T, (num_states, self.q**subsyst_size, -1))
        # find singular values to get Schmidt decomposition
        # if there are many states then svd is computed for last two indices (correct)
        svals = np.linalg.svd(state_mat, compute_uv=False)
        # make all singular values slightly positive to take log
        return svals**2 if num_states > 1 else (svals**2)[0]

    # gives IPR of one vector or of a matrix of column vectors
    # if vector is in sub-sector it gives IPR within that sub-sector
    def getIPR(self, state, q=2):
        return np.sum(np.abs(state)**(2*q), axis=0)

    def getIPR_LR(self, stateL, stateR, q=2):
        return np.sum(np.abs(stateL.conj() * stateR)**q, axis=0)

    # gives list of local spins <Sz> on each lattice site (which is index)
    # for input of many states, output is indexed: row = state, col=positions
    def getLocalSpin(self, state):
        positions = []
        sz = self.spin_mat_list[1]
        for i in range(self.L):
            sz_i = self.op_at_sites([sz], [i])
            if self.projector != None:
                sz_i = self.projector @ sz_i @ self.projector.T
            eval_i = np.einsum('...a,ab,b...->...', state.conj().T, sz_i.toarray(), state).real
            positions.append(eval_i)
        return np.array(positions).T  

    # local expecation value of generic operator in one or many states
    def getExpectationValue(self, state, op_list, site_list):
        op = self.op_at_sites(op_list, site_list)
        if self.projector != None:
            op = self.projector @ op @ self.projector.T
        expval = np.einsum('...a,ab,b...->...', state.conj().T, op.toarray(), state).real
        return expval

    # returns ground state of Hermitian Hamiltonian
    def getGroundState(self):
        if self.Ham.shape[0] < 2**10:
            vals, vecs = np.linalg.eigh(self.Ham.toarray())
        else:
            vals, vecs = sparse.linalg.eigsh(self.Ham, k=6, which='SA')
        vals, vecs = vals[np.argsort(vals)], vecs[:, np.argsort(vals)] 

        return vals[0], vecs[:,0]

    # returns number of krylov spaces and a list of their respective sizes
    # optionally returns indices of all basis elements in each subspace (list of lists)
    # also works if we are within an overall symmetry sector
    def getKrylov(self, states=False):
        # number of sectors and list of sector labels (0 to num_Krylov-1) for basis states
        num_Krylov, labels = sparse.csgraph.connected_components(abs(self.Ham))
        # sizes of the different sectors
        sizes = np.unique(labels, return_counts=True)[1]
        # Make relevant labels range from 1 to num_Krylov (label of 0 = outside symm sector)
        labels += 1
        # project back to full Hilbert space if we are in symm sector
        # also keep track of how many states are outside of overall symmetry sector
        num_outside = 0
        if self.projector != None:
            labels = self.projector.T @ labels
            num_outside = np.diff(self.projector.shape)[0]
        
        if states == True:
            # make list of ordered pairs (state_index, sector_label)
            labels = np.column_stack( (np.arange(self.q**self.L), labels) )
            # sort by increasing sector label
            labels = labels[labels[:,1].argsort()]
            # skip states that are outside the overall symmetry sector 
            labels = labels[num_outside:, :]
            
            # make list of lists: each list is group of states in same Krylov sector
            sector_inds = []
            count = 0
            for size in sizes:
                # collect state indices that have same sector label
                sector_inds.append(labels[count:count+size, 0])
                count += size
            
            # sort lists of sizes and sector basis elements from largest to smallest sectors
            sector_inds.sort(key=len, reverse=True)
            sizes = np.sort(sizes)[::-1]
            return [num_Krylov, sizes, sector_inds]
        else:
            # sort list of sizes from largest to smallest before returning
            sizes = np.sort(sizes)[::-1]
            return [num_Krylov, sizes]

    # returns exponentiated Ham to give time evolution operator e^(-iHt)
    def getU(self, time):
        U = sparse.linalg.expm(-1.j*time*self.Ham.tocsc())
        return U

    # time evolves state by applying U more efficiently
    # also works for non-Hermitian systems
    def timeEvolve(self, state, time):
        return sparse.linalg.expm_multiply(-1.j*time*self.Ham.tocsc(), state)



