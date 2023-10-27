# SpinChainED
Exact diagonalization tools for one-dimensional spin chains

The basic class in this repo is SpinChain, which instantiates a one-dimensional spin chain with set spin, length, and boundary conditions. Users can then specify a Hamiltonian to study the energies and states of the spin chain. Additional features include the ability to project the Hamiltonian into symmetry subsectors (of fixed magnetization and dipole moment), to apply time evolution to an initial state with the Hamiltonian, calculate expectation values of local/global operators, calculate entanglement measures, and study the sectors of the Hilbert space connected by the Hamiltonian.
