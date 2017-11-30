from hf import *
import numpy as np
import copy

def generate_basisfunctions(molecule, basis_set):
    basis_functions = []
    for atom in molecule:
        if not atom.atomic_number in basis_set:
            print(atom.atomic_number)
            raise
        for orbital in basis_set[atom.atomic_number]:
            if orbital[0] == 'S':
                basis_functions.append( ContractedGTO((0,0,0), atom.pos, orbital[1], orbital[2]) )
            elif orbital[0] == 'P':
                basis_functions.append( ContractedGTO((1,0,0), atom.pos, orbital[1], orbital[2]) )
                basis_functions.append( ContractedGTO((0,1,0), atom.pos, orbital[1], orbital[2]) )
                basis_functions.append( ContractedGTO((0,0,1), atom.pos, orbital[1], orbital[2]) )
            else:
                raise "Not Supported"
    return basis_functions

def perturbate_system(atom, index, displacement):
    ret = []
    for i in range(len(atom)):
        if i != index:
            ret.append( atom[i] )
        else:
            ret.append( Nuclear(atom[i].pos + displacement, atom[i].atomic_number) )
    return ret

def grad(nelec, basis_functions, atom, dr = 0.05):
    if dr < 0: raise
    #e = rhf(nelc, basis_functions, atom)
    ret = []
    for i in range(len(atom)):
        system = perturbate_system(atom, i, np.array([+dr, 0, 0]))
        basis  = generate_basisfunctions(system, basis_functions)
        pdx = rhf(nelec, basis, system)
        del system, basis

        system = perturbate_system(atom, i, np.array([-dr, 0, 0]))
        basis  = generate_basisfunctions(system, basis_functions)
        mdx = rhf(nelec, basis, system)
        del system, basis

        system = perturbate_system(atom, i, np.array([0, +dr, 0]))
        basis  = generate_basisfunctions(system, basis_functions)
        pdy = rhf(nelec, basis, system)
        del system, basis

        system = perturbate_system(atom, i, np.array([0, -dr, 0]))
        basis  = generate_basisfunctions(system, basis_functions)
        mdy = rhf(nelec, basis, system)
        del system, basis

        system = perturbate_system(atom, i, np.array([0, 0, +dr]))
        basis  = generate_basisfunctions(system, basis_functions)
        pdz = rhf(nelec, basis, system)
        del system, basis

        system = perturbate_system(atom, i, np.array([0, 0, -dr]))
        basis  = generate_basisfunctions(system, basis_functions)
        mdz = rhf(nelec, basis, system)
        del system, basis

        dEdx = (pdx-mdx)/(2*dr)
        dEdy = (pdy-mdy)/(2*dr)
        dEdz = (pdz-mdz)/(2*dr)
        ret.append( np.array([dEdx, dEdy, dEdz]) )
    return ret

def update_system(atom, force, scale):
    # Tiny Steepest Descent Algorithm
    new_system = []
    for i in range(len(atom)):
        displacement = -1.0 * scale * force[i] 
        new_system.append( Nuclear(atom[i].pos + displacement, atom[i].atomic_number) )
    return new_system

def optimize(nelec, basis_functions, atom, max_step = 10, dr_for_grad = 0.05, scale_for_displacement = 0.8, converge_criteria = 0.0001):
    system = atom
    energy_history = []
    optimized_flag = False
    for n_step in range(max_step):
        print("Optimization step:{}".format(n_step))
        for i in range(len(system)):
            print("{}  {}".format(system[i].atomic_number, system[i].pos))
        bfs = generate_basisfunctions(system, basis_functions)
        energy_history.append( rhf(nelec, bfs, system) )

        if 1 <= n_step and abs(energy_history[-1]-energy_history[-2]) < converge_criteria:
            optimized_flag = True
            break

        force = grad(nelec, basis_functions, system, dr_for_grad)
        # Determine the Displcement Vector for each atoms
        system = update_system(system, force, scale_for_displacement)

    if optimized_flag == True:
        print("Optimize Done")
        print("Energies: ", energy_history)
    else:
        print("Optimized Failed")
    return (optimized_flag, system)
            

# Degine STO-3G system


if __name__ == "__main__":
    basis_set = dict()
    basis_set[1] = [ ('S', [0.168856, 0.623913, 3.42525], [0.444635, 0.535328, 0.154329]) ]
    basis_set[2]= [ ('S', [0.31364979, 1.15892300, 6.36242139], [0.44463454, 0.53532814, 0.15432897]) ]
    basis_set[8] = [
            ('S', [130.7093200,23.8088610,6.4436083], [0.15432897, 0.53532814, 0.44463454] ),
            ('S', [5.0331513,1.1695961,0.3803890], [-0.09996723, 0.39951283, 0.70011547] ),
            ('P', [5.0331513,1.1695961,0.3803890],[ 0.15591627, 0.60768372,  0.39195739] )  ]

    h2_system = []
    h2_system.append(Nuclear([0  ,0,0], 1))
    h2_system.append(Nuclear([2.0,0,0], 1))
    basis_array_H2 = generate_basisfunctions(h2_system, basis_set)
    (optimized_flag, optimized_system) = optimize(2, basis_set, h2_system, scale_for_displacement = 1.0)
    print("Optimized Coordinate:")
    for i in range(len(optimized_system)):
        print("{} : {}".format(optimized_system[i].atomic_number, optimized_system[i].pos))
