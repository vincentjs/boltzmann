# lbm.jl
# Lattice Boltzmann Method (D2Q9) written in Julia
# Vincent San Miguel

# Rectangular domain dimensions, x₁ in East and x₂ in North
x₁ = 300
x₂ = 100

# Viscosity
ν = 0.02
# Relaxation parameter
ω = 1 / (3 * ν + 0.5)
# Inlet velocity
u₀ = 0.1

# For a 2D gas, the continuous distribution of the thermal velocities, v, 
# is given by the Boltzmann distribution
#
# Lattice is D2Q9. We specify 9 velocity vectors e₀ through e₈
# corresponding to zero, the 4 cardinal directions, and 4 diagonal directions.
# We can attach probabilities to these vectors. Since the fluid is at rest,
# the probability distribution is symmetrical, and velocities with equal
# magnitude - regardless of direction - will have equal weightings.
#
# Lattice total velocities, e = Δx / Δt
# e₀ = [0, 0], e₁ = [1, 0] (E), e₂ = [0, 1] (N), e₃ = [-1, 0] (W),
# e₄ = [0, -1] (S), e₅ = [1, 1] (NE), e₆ = [-1, 1] (NW), e₇ = [-1, -1] (SW),
# e₈ = [1, -1] (SE)
#
# We can then choose probability weightings, w, to approximate the
# continuous Boltzmann distribution such that these average velocity
# values are preserved

# Lattice probability weightings
w₀ = 4.0 / 9.0
w₁ = w₂ = w₃ = w₄ = 1.0 / 9.0
w₅ = w₆ = w₇ = w₈ = 1.0 / 36.0

# We can obtain number densities for a fluid in motion by rewriting the
# continuous Boltzmann distribution, replacing the thermal velocity with
# the scaled total velocity minus the macroscopic velocity, v = e * c - u,
# and taking the Taylor series

# Initial number densities - solution domain initialized with steady
# rightward flow
n₀ = w₀ * ones(x₂, x₁) - 1.5 * u₀^2
n₁ = w₁ * ones(x₂, x₁) + 3.0 * u₀ + 4.5 * u₀^2 - 1.5 * u₀^2
n₂ = w₂ * ones(x₂, x₁) - 1.5 * u₀^2
n₃ = w₃ * ones(x₂, x₁) - 3.0 * u₀ + 4.5 * u₀^2 - 1.5 * u₀^2
n₄ = w₄ * ones(x₂, x₁) - 1.5 * u₀^2
n₅ = w₅ * ones(x₂, x₁) + 3.0 * u₀ + 4.5 * u₀^2 - 1.5 * u₀^2
n₆ = w₆ * ones(x₂, x₁) - 3.0 * u₀ + 4.5 * u₀^2 - 1.5 * u₀^2
n₇ = w₇ * ones(x₂, x₁) - 3.0 * u₀ + 4.5 * u₀^2 - 1.5 * u₀^2
n₈ = w₈ * ones(x₂, x₁) + 3.0 * u₀ + 4.5 * u₀^2 - 1.5 * u₀^2

# Macroscopic density
ρ = n₀ + n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ + n₈

# Macroscopic x-velocity and y-velocity
ux = ((n₁ + n₅ + n₈) - (n₃ + n₆ + n₇)) / ρ
uy = ((n₂ + n₅ + n₆) - (n₄ + n₇ + n₈)) / ρ

# Initialize barriers
barrier = falses(x₂, x₁)

barrier[convert(Int, 0.5 * x₂ - 8) : convert(Int, 0.5 * x₂ + 8),
        convert(Int, 0.5 * x₂)] = true

# Sites around barrier
b₁ = circshift(barrier, [0, 1])
b₂ = circshift(barrier, [1, 0])
b₃ = circshift(barrier, [0, -1])
b₄ = circshift(barrier, [-1, 0])
b₅ = circshift(barrier, [1, 1])
b₆ = circshift(barrier, [1, -1])
b₇ = circshift(barrier, [-1, -1])
b₈ = circshift(barrier, [-1, 1])

function collide!()
    # Collides all particles

    rho = n₀ + n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ + n₈

    ux = ((n₁ + n₅ + n₈) - (n₃ + n₆ + n₇)) / ρ
    uy = ((n₂ + n₅ + n₆) - (n₄ + n₇ + n₈)) / ρ

    ux² = ux^2
    uy² = uy^2
    u² = ux² * uy²
    uxuy = ux * uy

    n₀ = (1 - ω) * n₀ + ω * w₀ * ρ * (1 - 1.5 * u²)
    n₁ = (1 - ω) * n₁ + ω * w₁ * ρ * (1 - 1.5 * u² + 3 * ux + 4.5 * ux^2)
    n₂ = (1 - ω) * n₂ + ω * w₂ * ρ * (1 - 1.5 * u² + 3 * uy + 4.5 * uy^2)
    n₃ = (1 - ω) * n₃ + ω * w₃ * ρ * (1 - 1.5 * u² - 3 * ux + 4.5 * ux^2)
    n₄ = (1 - ω) * n₄ + ω * w₄ * ρ * (1 - 1.5 * u² - 3 * uy + 4.5 * uy^2)
    n₅ = (1 - ω) * n₅ + ω * w₅ * ρ * (1 - 1.5 * u² + 3 * (ux + uy) +
                                      4.5 * (u² + 2 * uxuy))
    n₆ = (1 - ω) * n₆ + ω * w₆ * ρ * (1 - 1.5 * u² + 3 * (uy - ux) +
                                      4.5 * (u² - 2 * uxuy))
    n₇ = (1 - ω) * n₇ + ω * w₇ * ρ * (1 - 1.5 * u² + 3 * (-ux - uy) +
                                      4.5 * (u² + 2 * uxuy))
    n₈ = (1 - ω) * n₈ + ω * w₈ * ρ * (1 - 1.5 * u² + 3 * (ux - uy) +
                                      4.5 * (u² - 2 * uxuy))

    # Force steady inlet flow
    fac = 3 * u₀ + 4.5 * u₀^2 - 1.5 * u₀^2

    n₁[:, 1] = w₁ * (1 + fac)
    n₃[:, 1] = w₃ * (1 - fac)
    n₅[:, 1] = w₅ * (1 + fac)
    n₆[:, 1] = w₆ * (1 - fac)
    n₇[:, 1] = w₇ * (1 - fac)
    n₈[:, 1] = w₈ * (1 + fac)

end

function stream!()
    # Move particles by one step in the direction of motion

    circshift!(n₁, [0, 1])
    circshift!(n₂, [1, 0])
    circshift!(n₃, [0, -1])
    circshift!(n₄, [-1, 0])
    circshift!(n₅, [1, 1])
    circshift!(n₆, [1, -1])
    circshift!(n₇, [-1, -1])
    circshift!(n₈, [-1, 1])

    # Handle barrier collisions
    n₁[b₁] = n₃[barrier]
    n₂[b₂] = n₄[barrier]
    n₃[b₃] = n₁[barrier]
    n₄[b₄] = n₂[barrier]
    n₅[b₅] = n₇[barrier]
    n₆[b₆] = n₈[barrier]
    n₇[b₇] = n₅[barrier]
    n₈[b₈] = n₆[barrier]

end

function curl(ux, uy)
    # Compute curl of macroscopic velocity

    return circshift(uy, [0, -1]) - circshift(uy, [0, 1]) -
           circshift(ux, [-1, 0]) + circshift(ux, [1, 0])

end

# Plot
Pkg.add("PyPlot")
using PyPlot

# Set Plot
fig = PyPlot.figure(figsize=(8,3))
fluidImg = PyPlot.imshow(curl(ux, uy),
                         origin="lower",
                         cmap=PyPlot.get_cmap("jet"),
                         interpolation="none")

# Set RGBA image
rgbaImg =  zeros(4, x₂, x₁)
# Color barrier
rgbaImg[barrier, 3] = 255
# Barrier image
barrierImg = PyPlot.imshow(rgbaImg, origin='lower', interpolation='none')

function nextFrame(frame)
    # Plot next frame

    for step = 1 : 20
        stream()
        collide()
    end

    fluidImg.set_array(curl(ux, uy))

    return fluidImg, barrierImg
end

animate = PyPlot.animation.FuncAnimation(fig, nextFrame, interval=1, blit=true)
PyPlot.show()

