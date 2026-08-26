# gen_input.py
import sys
import numpy as np

# Simulation parameters shared with the C, CUDA, and HIP implementations.
TOTAL_TIME = 20.0
DT = 0.01
DUMP_INTERVAL = 10

# Gravitational constant used by the simulation.
G = 1.0

# Radial range for sampled positions.
R_MIN = 5.0
R_MAX = 50.0

# Body-mass range.
M_MIN = 0.5
M_MAX = 5.0

# Velocity perturbation scale.
VEL_NOISE = 0.2

# Set to zero to omit the central body.
CENTRAL_MASS = 500.0


def sample_mass():
    return np.random.uniform(M_MIN, M_MAX)


def random_unit_vector():
    """Sample a direction uniformly on the three-dimensional unit sphere."""
    z = 2.0 * np.random.rand() - 1.0  # cos(theta) in [-1,1]
    t = 2.0 * np.pi * np.random.rand()  # phi in [0, 2pi)
    r_xy = np.sqrt(1.0 - z * z)
    x = r_xy * np.cos(t)
    y = r_xy * np.sin(t)
    return np.array([x, y, z], dtype=float)


def gen_random3d(N, filename):
    bodies = []

    # Place the optional central body at the origin.
    if CENTRAL_MASS > 0.0:
        bodies.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, CENTRAL_MASS])

    # Sample N orbiting bodies.
    for _ in range(N):
        # Cube-root sampling gives a uniform volume distribution.
        u = np.random.rand()
        r = np.cbrt(u) * (R_MAX - R_MIN) + R_MIN

        # Sample the radial direction.
        dir_r = random_unit_vector()
        pos = r * dir_r
        x, y, z = pos

        m = sample_mass()

        # Approximate a circular-orbit tangent around the central body.
        if CENTRAL_MASS > 0.0:
            # Estimate circular velocity from the central mass.
            v_c = np.sqrt(G * CENTRAL_MASS / r)

            # Construct a tangent from a non-collinear direction.
            tmp = random_unit_vector()
            if abs(np.dot(tmp, dir_r)) > 0.9:
                tmp = random_unit_vector()
            # The cross product is perpendicular to the radial direction.
            v_dir = np.cross(dir_r, tmp)
            v_dir /= np.linalg.norm(v_dir)

            v = v_c * v_dir

            # Perturb the idealized circular velocity.
            v += VEL_NOISE * np.random.randn(3)
        else:
            # Without a central body, use a small random velocity.
            v = VEL_NOISE * np.random.randn(3)

        vx, vy, vz = v

        bodies.append([x, y, z, vx, vy, vz, m])

    N_total = len(bodies)

    with open(filename, "w") as f:
        f.write(f"{N_total}\n")
        f.write(f"{TOTAL_TIME} {DT} {DUMP_INTERVAL}\n")
        for b in bodies:
            f.write(
                "{:.6f} {:.6f} {:.6f} " "{:.6f} {:.6f} {:.6f} " "{:.6f}\n".format(*b)
            )

    print(f"Generated {N_total} bodies -> {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_input_random3d.py N output.txt")
    else:
        N = int(sys.argv[1])
        out = sys.argv[2]
        gen_random3d(N, out)
