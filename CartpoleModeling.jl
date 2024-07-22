using DifferentialEquations
using Plots

# Define the parameters
const M = 0.5  # Mass of the cart
const m = 0.2  # Mass of the pendulum
const l = 0.3  # Length of the pendulum
const g = 9.81 # Gravitational acceleration
const b = 0.1  # Friction coefficient
const I = 1 / 3 * m * l^2  # Moment of inertia of the pendulum

function step(t)
  return t < 1.0 ? 0.0 : 1.0
end

function nonlinear_dyn(x, u)
  _, xₜ, θ, θₜ = x

  β = I * (M + m) + M * m * l^2 + sin(θ)^2 * m^2 * l^2

  return [
    xₜ,
    (
      (I + m * l^2) * (m * l * θₜ^2 * sin(θ) - b * xₜ)
      -
      m^2 * l^2 * g * cos(θ) * sin(θ)
      +
      (I + m * l^2) * u
    ) / β,
    θₜ,
    (
      m * l * cos(θ) * (b * xₜ - m * l * θₜ^2 * sin(θ))
      +
      (M + m) * m * l * g * sin(θ)
      -
      m * l * cos(θ) * u
    ) / β
  ]
end

function solve_dynamics!(du, u, p, t)
  F = step(t)
  du .= f(u, F)
end

function euler(x, u, dt, f)
  return x + f(x, u) * dt
end

function runge_kutta(x, u, h, f)
  k1 = f(x, u)
  k2 = f(x + h / 2 * k1, u)
  k3 = f(x + h / 2 * k2, u)
  k4 = f(x + h * k3, u)

  return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
end

function solve_discrete_dyn(x0, t, dt, f; method="euler")
  u = step
  states = [x0]
  x = x0
  for tk in t[1:end-1]
    x = if method == "runge_kutta"
      runge_kutta(x, u(tk), dt, f)
    else
      euler(x, u(tk), dt, f)
    end
    push!(states, x)
  end
  return reduce(hcat, states)'
end

f = nonlinear_dyn

# Initial conditions
u0 = [0.0, 0.0, 0.0, 0.0]  # [x, dx, theta, dtheta]

tspan = (0.0, 50.0)

prob = ODEProblem(solve_dynamics!, u0, tspan)

dt = 0.01
t = range(tspan[1], tspan[2], step=dt)

sol = solve(prob, saveat=t)
sol2 = solve_discrete_dyn(u0, t, dt, f, method="runge_kutta")
sol3 = solve_discrete_dyn(u0, t, dt, f)

# Plot the results
plot(t, sol2[:, 1], label="Nonlinear RK4")
plot!(t, sol3[:, 1], label="Nonlinear Euler")
plot!(t, sol[1, :], label="Nonlinear", show=true)
title!("Cart position")
xlabel!("t [s]")
ylabel!("x [m]")
readline()
