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

modelname = "potts3";

J = 1.0; hV = 1.0; hU = 0.0
w_U, w_V, w_om = ED.weylops(3)
ham_comp = ED.ham_vpotts_comp
ham = ED.ham_vpotts
h = reshape(ham(3, J, hV, hU), (3,3,3,3))
h_comp = ham_comp(3, J, hV, hU)

get_Pchg = (chg::Int) -> begin 
    #Projects onto the chg sector
    (s, tmp1, tmp2, s_sz)->begin
        copy!(tmp2, s) #copy original state
        
        ED.apply_global_prod_op!(tmp2, w_V, tmp=tmp1, s_sz=s_sz)
        
        copy!(tmp1, s) #copy original state again
        LinAlg.BLAS.axpy!(w_om^chg, tmp2, s) #now overwrite it with |s> + om^chg * V|s>
        
        ED.apply_global_prod_op!(tmp1, w_V', tmp=tmp2, s_sz=s_sz)
        
        LinAlg.BLAS.axpy!(w_om^(-chg), tmp1, s) #|s> + om^chg * V|s> + om^-chg * V'|s>

        scale!(s, 1/3) #1/3(|s> + om^chg * V|s> + om^-chg * V'|s>)
        s
    end
end

get_Pmom_Pchg = (ps::Vector{Float64}, chg::Int) -> begin
    Pmom = ED.get_Pmoms(ps)
    Pchg = get_Pchg(chg)
    (s, tmp1, tmp2, s_sz)->begin
        s = Pchg(s, tmp1, tmp2, s_sz)
        s = Pmom(s, tmp1, tmp2, s_sz)
        s
    end
end

datapath = "data/"

#Charge 0, p up to +/- 3
nev = 158

eV_data = Dict{Int,Any}()
ed_data = Dict{Int,Any}()
Hs_data = Dict{Int,Any}()

ED.ed!(eV_data, collect(4:7), h, nev, symm_func=get_Pchg(0));
ED.ed!(eV_data, collect(8:14), h, nev, symm_func=get_Pmom_Pchg([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0], 0));

save("$(datapath)$(modelname)_redP_chg0_eV.jld", "eV", eV_data)

for (N, eVd) in eV_data
    ed_data[N] = (eVd[1:2]...)
end

Hs_data = Dict{Int,Any}()
LCFT.Hns_ed!(Hs_data, eV_data, h_comp, 3, 2)

LCFT.normalise_Hn!(Hs_data, ed_data)

save("$(datapath)$(modelname)_redP_chg0_LCFTdata.jld", "ev", ed_data, "Hs", Hs_data)