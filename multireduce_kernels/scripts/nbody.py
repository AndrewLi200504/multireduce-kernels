# The following code benchmarks torch and mk on a real life application: finding the 
# net force exerted by electrostatic and gravitational forces on a single target particle,
# given the masses, charges, and distances between the particles and the target particle.

import torch
import multireduce_kernels as mk
M = 1.67e-27
E = 1.602e-19
G = 6.67e-11
K = 9.0e9

def nbody(): 
    return


def main():
    return 

if __name__ == "__main__": 
    bodies = torch.randint(low=1, high=5, size=(10000000, 3))
    bodies = bodies[:, 0] * M
    main()