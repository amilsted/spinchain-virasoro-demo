warn("This script requires a large amount of memory to run!")

#BLAS.set_num_threads(6)

using TensorOperations
using JLD
using LinearMaps

include("src/PlotTools.jl")

include("src/EDutils.jl")
import EDutils
ED = EDutils

include("src/LatCFT.jl")
import LatCFT
LCFT = LatCFT

modelname = "ANNNI3";

ham_comp = (λ, hz, hx, Δ, Δd) -> begin
    h1, h2 = ED.ham_ising_comp(λ, hz, hx)
    h2 = h2 + Δd * kron(ED.pauliZ, ED.pauliZ)
    h3 = Δ * kron(ED.pauliX, eye(2), ED.pauliX)
    h1,h2,h3
end

ham = (λ, hz, hx, Δ, Δd) -> begin
    h1, h2, h3 = ham_comp(λ, hz, hx, Δ, Δd)
    h3 + 0.5(kron(h2, eye(2)) + kron(eye(2), h2)) + 1/3 * (kron(h1, eye(4)) + kron(eye(2), h1, eye(2)) + kron(eye(4), h1))
end

λ = 1.0; hz = 1.0; Δ = -0.5; Δd = -0.5 #Self-dual ANNNI model remains at Ising criticality for ferromagnetic interactions.
h = reshape(ham(λ, hz, 0.0, Δ, Δd), (2,2,2,2,2,2))
h_comp = ham_comp(λ, hz, 0.0, Δ, Δd)

nev = 80

eV_data = Dict{Int,Any}()
ed_data = Dict{Int,Any}()
Hs_data = Dict{Int,Any}()

datapath = "data/"

ED.ed!(eV_data, collect(6:1:8), h, nev)

ps_allowed = Float64[-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0]
ED.ed!(eV_data, collect(9:1:22), h, nev, symm_func=ED.get_Pmoms(ps_allowed))

save("$(datapath)$(modelname)_redP_eV.jld", "eV", eV_data)

for (N, eVd) in eV_data
    ed_data[N] = (eVd[1:2]...)
end

Hs_data = Dict{Int,Any}()
LCFT.Hns_ed!(Hs_data, eV_data, h_comp, 2, 2)

LCFT.normalise_Hn!(Hs_data, ed_data)

save("$(datapath)$(modelname)_redP_LCFTdata.jld", "ev", ed_data, "Hs", Hs_data)