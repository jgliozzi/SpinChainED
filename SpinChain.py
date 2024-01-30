# Class for Exact Diagonalization Calculations
import numpy as np
from scipy import sparse, linalg, special
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
        sm = np.diag([np.sqrt(spin*(spin+1) - m*(m+1)) 
                    for m in np.arange(-spin, spin, 1.0)], k=1) 
        sp = np.diag([np.sqrt(spin*(spin+1) - m*(m+1)) 
                    for m in np.arange(-spin, spin, 1.0)], k=-1) 
        self.spin_mat_list = [s0, sz, sp, sm] # spin matrices

        # Possible boundary conditions (bc) are 0 (open), 1 (periodic), or -1 (antiperiodic)
        # flux is an additional twist (multiplicative phase) applied to the boundary condtions
        self.bc = bc * np.exp(1.j*flux)

        # keep track of projector if we project to a sector of the Hilbert space
        self.projector = None
        # keep track of full hamiltonian if we project to a sector of the Hilbert space
        self.FullHam = None

    # returns identity and spin matrices (various forms allowed)
    # note that spin matrices by default include factor of spin (1/2 for spin-1/2, etc.)
    def getPauli(self, basis="pm"):
        s0, sz, sp, sm = self.spin_mat_list
        basis_dict = {
            "pm": [s0, sz, sp, sm], # Id, Sz, Sp, Sm (raising and lowering)
            "xyz": [s0, 1/2 * (sp + sm), 1/(2j) * (sp - sm), sz], # Id, Sx, Sy, Sz (standard)
            "pauli": [np.eye(2), np.array([[0, 1], [1, 0]]), 
                        np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])], # Id, Sigmax, Sigmay, Sigmaz (2 x 2)
            "clock_shift": [np.eye(self.q), np.diag([np.exp(2*np.pi*1j * n / self.q) for n in range(self.q)]), 
                                np.roll(np.eye(self.q), -1, axis=0)] # Id, Z (clock), X (shift) 
        }
        return basis_dict.get(basis, [])

    # returns Hamiltonian (which is a sparse array)
    def getHam(self):
        return self.Ham

    # return dimensions of Hilbert space (or subspace if Hamiltonian is projected)
    def getHilbertDim(self):
        return self.Ham.shape[0]

    # returns array of local spin/occupation lists assoc w/ basis elements
    # (e.g., element 0 -> |00000>, element 1 -> |00001>, element 2 -> |00011>)
    # row index = basis element number, col index = site number, entry = spin
    def basis_to_occ(self):
        # much faster if we can use binary operations
        q, L = self.q, self.L
        if q==2:
            # make column vector of basis element indices
            basis_arr = np.arange(2**L).reshape([-1,1])
            # make row vector of powers of two (increasing R to L)
            mask = 2**np.arange(L)[::-1]
            # apply bitwise AND between elements of basis_arr and mask
            # (e.g. 5 & 4 -> 101 & 100 = 100 -> 4)
            # each row is now like [5&8, 5&4, 5&2, 5&1] = [0, 4, 0, 1]
            # converting nonzero entries to 1 gives list of binary digits of 5 -> [0,1,0,0] 
            occ_list = (basis_arr & mask).astype(bool).astype("int8")
        # slightly slower for spin-1 and beyond, use numba to speed up
        else:
            @jit(nopython=True, cache=True)
            def return_answer():
                basis_arr = np.arange(q**L).reshape((-1,1))
                powers_of_q = q**np.arange(L)[::-1]
                return (basis_arr // powers_of_q % q)
            occ_list = return_answer()
        # subtract total spin to set range as (-spin, spin) instead of (0, 2*spin) 
        return occ_list - self.spin

    # returns magnetization and or dipole of all basis elements
    def mag_dip_basis_list(self, mag=True, dip=True):
        bto = self.basis_to_occ()
        if mag and dip:
            mag_of_basis = np.sum(bto, axis=1)
            dip_of_basis = np.einsum('ij,j->i', bto, np.arange(self.L))
            return (mag_of_basis, dip_of_basis)
        elif mag:
            return np.sum(bto, axis=1)
        elif dip:
            return np.einsum('ij,j->i', bto, np.arange(self.L))
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
    # site_lists that wrap around chain are ignored if bc = 0
    def setHamTerm(self, op_list, site_list_list, coeff=1, ignore_bc=False):
        for site_list in site_list_list:
            site_list = np.array(site_list) % self.L          
            # site in site_list that first wraps around chain
            wrap_index = np.where(np.diff(site_list) < 0)[0]
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

    # projector onto fixed magnetization sector
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

    # projector onto fixed dipole sector
    # include option to define dipole moment periodically mod L (valid for PBC)
    def P_dip(self, Pdip, periodic_dip=True):
        # list of dipoles of basis states
        dip_list = self.mag_dip_basis_list(mag=False, dip=True)
        # indices of basis states in desired sector
        # define dipole moment mod L for periodic boundary conditions
        if periodic_dip:
            indices = np.where(np.abs((dip_list - Pdip)%self.L) < 1e-9)[0]
        else:
            indices = np.where(np.abs(dip_list - Pdip) < 1e-9)[0]        

        P = sparse.lil_array((len(indices), self.q**self.L))
        for i in range(len(indices)):
            P[i, indices[i]] = 1.0
        return P

    # projector onto fixed magnetization and dipole sector
    # include option to define dipole moment periodically mod L (valid for PBC)
    def P_mag_dip(self, M, Pdip, periodic_dip=False):
        # list of magnetizations and dipole moments of basis elements
        mag_list, dip_list = self.mag_dip_basis_list(mag=True, dip=True)
        # indices of basis states in desired sector
        # project to Pdip (mod L) if we have PBC because shifting dipole is only well-defined mod L (and with M=0)
        if periodic_dip:
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

    # Entanglement entropy of region (specify as list of sites)
    # works for single state or multiple states (inputted as columns of an array)
    # von Neumann entropy by default (n=1), Rényi if n != 1
    # optionally return the full entanglement spectrum instead
    def getEntEntropy(self, state, region, n=1, ent_spectrum=False):
        # put state(s) into full Hilbert space
        num_states = 1 if len(state.shape)==1 else state.T.shape[0]
        if self.projector is not None:
            state = self.projector.T @ state

        # reshape to num_states x q x q x .... x q (one axis of length q per site)
        psi = np.reshape(state.T, [num_states] + self.L * [self.q])
        # for backwards compatibility w/ previous versions, allow int x to refere to region [0, x]
        if type(region) == int:
            region = range(region)
        regionB = [k+1 for k in range(self.L) if k not in region] # region to trace out
        regionA = [k+1 for k in region] # region to keep

        # reshape into matrices separating |psi> = ∑ w_{ij} |i>_{region A} |j>_{region B}, w_{ij} ~ sqrt(|psi><psi|)
        state_mat = np.reshape(np.transpose(psi, [0] + regionA + regionB), (num_states, self.q**len(regionA), -1))
        svals = np.linalg.svd(state_mat, compute_uv=False)
        
        if ent_spectrum:
            return svals**2 if num_states > 1 else (svals**2)[0]
        else:
            if n == 1:
                entropy = -np.sum(svals**2 * np.log(svals**2 + 1e-20), axis=1) # von Neumann S
            else:
                entropy = np.log(np.sum(svals**(2*n), axis=1)) / (1 - n) # Renyi S_n    

            return entropy if num_states > 1 else entropy[0]

    # get mutual information between two regions (A and B are lists)
    # works for multiple states and optionally renyi index
    def getMutualInfo(self, state, regionA, regionB, n=1):
        S_A = self.getEntEntropy(state, regionA, n)
        S_B = self.getEntEntropy(state, regionB, n)
        S_AB = self.getEntEntropy(state, regionA+regionB, n)
        return (S_A + S_B - S_AB)

    # One dimensional (SPT and SSB) version of topological entanglement entropy
    # Xiao-Gang Wen definition, split chain into A, B, D, C subregions
    # currently only works for a single state !
    # S_AB + S_BC - S_B - S_ABC
    def getStopo(self, state):
        L = self.L
        A, B, D, C = [*range(0, L//4)], [*range(L//4, L//2)], [*range(L//2, 3*L//4)], [*range(3*L//4, L)]
        
        S_AB = self.getEntEntropy(state, A+B)
        S_BC = self.getEntEntropy(state, B+C)
        S_B = self.getEntEntropy(state, B)
        S_ABC = self.getEntEntropy(state, A+B+C)

        return (S_AB + S_BC - S_B - S_ABC)

    # Reduced density matrix: only works for a single state
    def getReducedDensityMat(self, state, region):
        # put state(s) into full Hilbert space
        if self.projector is not None:
            state = self.projector.T @ state

        # reshape to q x q x .... x q (one axis of length q (onsite hilbert space dimension) per site)
        psi = np.reshape(state.T, self.L * [self.q])
        # for backwards compatibility w/ previous versions, allow int x to refer to region [0, x]
        if type(region) == int:
            region = range(region)
        regionB = [k for k in range(self.L) if k not in region] # region to trace out
        regionA = [k for k in region] # region to keep

        # reshape into matrices separating |psi> = ∑ w_{ij} |i>_{region A} |j>_{region B}, w_{ij} ~ sqrt(|psi><psi|)
        state_mat = np.reshape(np.transpose(psi, regionA + regionB), (self.q**len(regionA), -1))
        # use this Schmidt decomposition to construct reduced density matrix in occupation basis
        rho = state_mat @ state_mat.conj().T

        return rho

    # gives IPR of one vector or of a matrix of column vectors
    # if vector is in sub-sector it gives IPR within that sub-sector
    def getIPR(self, state, q=2):
        return np.sum(np.abs(state)**(2*q), axis=0)

    def getIPR_LR(self, stateL, stateR, q=2):
        return np.sum(np.abs(stateL.conj() * stateR)**q, axis=0)

    # local expecation value of generic operator in one or many states (columns)
    def getExpectationValue(self, state, op_list, site_list):
        num_states = 1 if len(state.shape)==1 else state.T.shape[0]

        op = self.op_at_sites(op_list, site_list)
        if self.projector is not None:
            op = self.projector @ op @ self.projector.T

        expval = state.conj().T @ op @ state
        if num_states > 1:
            expval = np.diag(expval)

        return expval.real

    # gives list of local spins <Sz> on each lattice site (which is index)
    # for input of many states (cols), output is indexed: row = state, col=positions
    def getLocalSpin(self, state):
        positions = []
        sz = self.spin_mat_list[1]
        
        for i in range(self.L):
            eval_i = self.getExpectationValue(state, [sz], [i])
            positions.append(eval_i)
        
        return np.array(positions).T  

    # returns ground state of Hermitian Hamiltonian
    def getGroundState(self):
        if self.Ham.shape[0] < 2**10:
            vals, vecs = np.linalg.eigh(self.Ham.toarray())
        else:
            vals, vecs = sparse.linalg.eigsh(self.Ham, k=1, which='SA')
        vals, vecs = vals[np.argsort(vals)], vecs[:, np.argsort(vals)] 

        return vals[0], vecs[:,0]

    # returns number of krylov spaces of Hamiltonian and a list of their respective sizes
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
        if self.projector is not None:
            labels = self.projector.T @ labels
            num_outside = np.diff(self.projector.shape)[0]
        
        if states:
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
        result = sparse.linalg.expm_multiply(-1.j*time*self.Ham.tocsc(), state)
        return result/np.linalg.norm(result)
    
    # Chebyshev approximation to time evolution at certain order (only Hermitian H)
    # optionally import E_min, E_max, and H_tilde (rescaled hamiltonian) if possible
    # optionally compute expectation value of operator w/o computing state
    def timeEvolveCheb(self, state, time, order, E_range=None, H_tilde=None, op_list=None, site_list=None):
        D = self.getHilbertDim()
        # import E_min and E_max of Hamiltonian if known, recalculate otherwise
        if E_range is None:
            E_min = sparse.linalg.eigsh(self.Ham, k=1, which='SA', return_eigenvectors=False)[0]
            E_max = sparse.linalg.eigsh(self.Ham, k=1, which='LA', return_eigenvectors=False)[0]
        else:
            E_min, E_max = E_range
        
        # rescaling factors
        eps = 1e-4
        a = (E_max - E_min)/(2 - eps)
        b = (E_max + E_min)/2

        # rescaled Hamiltonian to have spectrum in [-1, 1] (import if possible)
        if H_tilde is None:
            H_tilde = (self.Ham - b * sparse.eye(D)) / a          
        H_tilde_op = sparse.linalg.LinearOperator((D, D), matvec=lambda x: H_tilde @ x)

        # set coefficients of Chebyshev expansion
        coeffs = 2 * (-1j)**np.arange(order) * special.jv(np.arange(order), a*time)
        coeffs[0] /= 2 
        
        # create states for Chebyshev expansion (these are column states)
        cheb_states = np.empty((D, order), dtype=complex)
        m0 = state
        m1 = H_tilde_op.matvec(m0)
        cheb_states[:, 0] = m0
        cheb_states[:, 1] = m1
        # recursively create rest of Chebyshev states
        for n in range(2, order):
            m_next = 2 * H_tilde_op.matvec(m1) - m0
            cheb_states[:, n] = m_next
            m0, m1 = m1, m_next
        
        # compute state |psi(t)>
        if op_list is None:
            result = np.einsum('i,ji->j', coeffs, cheb_states) 
            result *= np.exp(-1.j * (E_min + a * (1-eps/2)) * time) 
            return result/np.linalg.norm(result)
        # compute expectation value of operator <psi(t)|O|psi(t)>
        else:
            op_full = self.op_at_sites(op_list, site_list) 
            coeff_mat = np.einsum('a,b -> ab', coeffs.conj(), coeffs)
            op_cheb = cheb_states.T.conj() @ op_full @ cheb_states
            return np.sum(coeff_mat * op_cheb).real

    # time evolves state using Krylov subspace method (only Hermitian H)
    # max_states is maximum size of Krylov subspace, tol sets convergence criterion
    # optionally compute expectation value of operator w/o computing state
    def timeEvolveKrylov(self, state, time, max_states, tol=1e-3, op_list=None, site_list=None):
        D = self.getHilbertDim()
        Ham_op = sparse.linalg.LinearOperator((D, D), matvec=lambda x: self.Ham @ x) # faster mat-vec product
        
        # Applies hamiltonian to krylov subspace (columns) and generates new vector 
        # returns orthonormalized updated krylov subspace with extra vector
        def applyH_ortho(Ham_op, prev_states):
            new_state = Ham_op.matvec(prev_states[:, -1]) # apply H
            orth_states, _ = np.linalg.qr( np.column_stack((prev_states, new_state)) ) # orthogonalize ALL states
            return orth_states

        # computes (forced) tridiagonal hamiltonian projected into krylov subspace
        # only works for Hermitian H ! (need to implement Arnoldi algorithm for non-herm)
        def compute_effectiveH(Heff, Ham_op, states):
            last_state = states[:, -1]
            penultimate_state = states[:, -2]
            
            ket = Ham_op.matvec(last_state) # apply H only to newest state

            alpha = np.dot(last_state.conj(), ket).real # diagonal component
            beta = np.dot(penultimate_state.conj(), ket).real # off_diagonal component

            # add 1 new diagonal and 1 super + sub diagonal elements
            Heff_new = np.pad(Heff, [(0, 1), (0, 1)], mode='constant')
            Heff_new[-1, -1] = alpha
            Heff_new[-1, -2], Heff_new[-2, -1] = beta, beta

            return Heff_new

        # Now perform the actual evolution
        v0 = state/np.linalg.norm(state)
        print(v0.shape)
        Heff = np.dot(v0, Ham_op.matvec(v0)).reshape((1,1)).real # first effective hamiltonian is 1 x 1
        krylov_states = v0.reshape((-1,1)) # make into column vector

        c_previous = np.array([1]) # compare successive iterations to see if state converges before max_states
        # iteratively generate krylov subspace
        for j in range(1, max_states):
            krylov_states = applyH_ortho(Ham_op, krylov_states) # D x (j+1) mat (j+1 vectors)
            Heff = compute_effectiveH(Heff, Ham_op, krylov_states) # (j+1) x (j+1) mat

            # unit vector in krylov basis
            e = np.zeros(j+1) 
            e[0] = 1 

            # time evolve state coefficients c in small Krylov basis
            if j < 90: 
                c = linalg.expm(-1.j*time*Heff) @ e
            else:
                c = sparse.linalg.expm_multiply(-1.j*time*sparse.csr_array(Heff), e)
            
            diff_coeff = c[:-1] - c_previous # compute Hilbert space distance between successive iterations
            distance = np.linalg.norm(diff_coeff)**2 + np.abs(c[-1])**2 # add newest coeff in separately
            if distance < tol:
                print(f"Converged using Krylov subspace of {j+1} states")
                break
            else:
                c_previous = c

        # compute state |psi(t)>
        if op_list is None:
             # project back into full Hilbert space to get final state
            state_final = np.linalg.norm(state) * np.einsum('i,ji->j', c, krylov_states)
            state_final /= np.linalg.norm(state_final)
            return state_final
        # compute expectation value of operator <psi(t)|O|psi(t)> 
        else:
            op_full = self.op_at_sites(op_list, site_list) 
            coeff_mat = np.einsum('a,b->ab', c.conj(), c) 
            op_krylov = krylov_states.T.conj() @ op_full @ krylov_states 
            return np.sum(coeff_mat * op_krylov).real

           


        
