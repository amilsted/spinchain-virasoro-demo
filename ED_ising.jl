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

modelname = "ising";

λ=1.0; hx = 0.0; hz = 1.0
ham_comp = ED.ham_ising_comp
ham = ED.ham_ising
h = reshape(ham(λ, hz, hx), (2,2,2,2))
h_comp = ham_comp(λ, hz, hx)

nev = 41

eV_data = Dict{Int,Any}()
ed_data = Dict{Int,Any}()
Hs_data = Dict{Int,Any}()

datapath = "data/"

ED.ed!(eV_data, collect(4:1:18), h, nev)

save("$(datapath)$(modelname)_eV.jld", "eV", eV_data)

for (N, eVd) in eV_data
    ed_data[N] = (eVd[1:2]...)
end

Hs_data = Dict{Int,Any}()
LCFT.Hns_ed!(Hs_data, eV_data, h_comp, 2, 2)

LCFT.normalise_Hn!(Hs_data, ed_data)

save("$(datapath)$(modelname)_LCFTdata.jld", "ev", ed_data, "Hs", Hs_data)