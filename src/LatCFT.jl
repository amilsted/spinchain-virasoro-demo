module LatCFT

using PyPlot
using PlotTools
using EDutils

function local_op_in_basis(lop::Vector{Array}, bvecsL::Vector{Array}, bvecsR::Vector{Array})
    @assert length(lop) == ndims(bvecsL[1])
    @assert length(lop) == ndims(bvecsR[1])
    
    op_new = zeros(eltype(bvecsL[1]), (length(bvecsL),length(bvecsR)))
    for j in 1:length(bvecsR)
        v = bvecsR[j]
        opv = EDutils.apply_local_op(vec(v), size(v), lop)
        for k in 1:length(bvecsL)
            w = bvecsL[k]
            op_new[k,j] = (vec(w)' * vec(opv))[1]
        end
    end
    op_new
end

function svgens(n::Int, Hcomp, normfac::Float64=1.0)
    N = length(Hcomp)
    d = size(Hcomp[1][1], 1) #assume all sites have same local dimension
    ns_max = div(maximum(map(ndims, Hcomp[1])), 2) #assume range of terms is the same for all sites
    
    L = Array[]
    for j in 1:N
        hs = Hcomp[j]
        Ln = 0.0
        for h in hs
            ns = div(ndims(h), 2)
            size(h, 1) == 0 && continue
            #Note: Recall that kron() argument ordering is the reverse of col-major index ordering
            hr = reshape(kron(eye(d^(ns_max-ns)), reshape(h, (d^ns, d^ns))), ntuple(x->d, 2ns_max))
            pos = j + ns/2 - 0.5
            Ln += cis(n * pos * 2π/N) * hr
        end
        push!(L, Ln)
    end
    L
end

function svgens{M}(n::Int, Hcomp::NTuple{M,Array}, Nsites::Int, normfac::Float64=1.0)
    hs = NTuple{M,Array}[Hcomp for n in 1:Nsites]
    svgens(n, hs, normfac)
end
                                
eV_site_decomp(eV::Matrix, d::Int, N::Int) = Array[reshape(eV[:,j], ntuple(x->d,N)) for j in 1:size(eV,2)]

function svgens_in_basis{M}(n::Int, Hcomp::NTuple{M,Matrix}, Nsites::Int, eVL::Matrix, eVR::Matrix, d::Int)
    eVL = eV_site_decomp(eVL, d, Nsites)
    eVR = eV_site_decomp(eVR, d, Nsites)
    
    Hcomp_r = ntuple(m->reshape(Hcomp[m], ntuple(x->d, 2m)), M) 
    
    Ln = svgens(n, Hcomp_r, Nsites)
    
    local_op_in_basis(Ln, eVL, eVR)
end

function Hns_ed(N::Number, evd, h_comp, d::Int, maxn::Int=2)
    en_pbc, mom_pbc, eV = evd[1:3]

    #note: shifting spectrum of H should not be necessary, since the modulation should cancel any identity contribution
    Matrix[svgens_in_basis(+n, h_comp, N, eV, eV, d) for n in 1:maxn]
end

function Hns_ed!(Hn_data, ev_data, h_comp, d::Int, maxn::Int=2; overwrite::Bool=true)
    for (N, evd) in ev_data
        !overwrite && haskey(Hn_data, N) && continue
        
        en_pbc, mom_pbc, eVR = evd[1:3]
        eVL = length(evd) >= 4 ? evd[4] : eVR

        Hn_data[N] = Hns_ed(N, evd, h_comp, d, maxn)
    end
    Hn_data
end

moms_Tev(Tev, N::Number, mom_res::Int) = angle(Tev.^mom_res)/π * N/(2mom_res)

function guess_Ts(H2)
    iT1 = indmax(abs(H2[:,1]))
    iT2 = indmax(abs(H2[1,:]))
    iT1,iT2
end

function normalising_factor(en, H2)
    iT1, iT2 = guess_Ts(H2)
    4.0 / (en[iT1] - en[1] + en[iT2] - en[1])
end

function normalising_factors{T,S1,S2}(ev_data::Dict{T,S1}, Hn_data::Dict{T,S2})
    fs = Dict{T,Float64}()
    for (k, Hns) in Hn_data
        fs[k] = normalising_factor(ev_data[k][1], Hns[2])
    end
    fs
end

function normalise_Hn!(Hn_data, ev_data)
    for (N, Hns) in Hn_data
        en = ev_data[N][1]
        fac = normalising_factor(en, Hns[2])
        @show N, fac
        for Hn in Hns
            scale!(Hn, fac)
        end
    end
end

function estimate_c(Hns)
    H2 = Hns[2]
    2 * (H2' * H2)[1,1]
end

function estimate_c_filterT(Hns)
    H2 = Hns[2]
    iT1, iT2 = guess_Ts(H2)
    2 * abs2(H2[iT1,1])
end

function plot_spec(ev_num::Int, N::Int, edata, op::Matrix; mom_res::Int=1, norm_fac=1.0, E0=nothing, c="r")
    en, Tev = edata[1:2]
    E0 = (E0 == nothing ? en[1] : E0)
    en = norm_fac * (en - E0)
    
    spins = moms_Tev(Tev, N, mom_res)
    ols = op[:,ev_num]
    
    plot(spins, en, "kx", markeredgewidth=1.0)
    scatter(spins, en, s=300*log10(100 * abs(ols)), alpha=0.8, edgecolors="None", color=c)
    scatter(spins[ev_num], en[ev_num], s=200, alpha=0.8, edgecolors=c, facecolors="none")
    xlabel(L"s")
    ylabel(L"\Delta")
    
    ols
end


function ols_low_high(op::Matrix, en::Vector, moms::Vector, tol::Float64=0.01; max_mom::Float64=Inf)
    lowen = Vector{Bool}[en .< en[j] for j in 1:length(en)]
    ols_lower = Float64[vecnorm(op[lowen[j],j]) for j in 1:length(en)]
    ols_higher = Float64[vecnorm(op[!lowen[j],j]) for j in 1:length(en)]
    inds = find(ol-> ol .<= tol, ols_lower)
    mominds = find(p-> abs(p) - sqrt(eps()) < max_mom, moms)
    inds = intersect(inds, mominds)
    ols_lower, ols_higher, inds
end

function qps(Hdata, en, moms, tol::Float64=0.01; max_mom::Float64=Inf)
    H1 = Hdata[1]
    op = (H1 + H1')/2
    ols_lower, ols_higher, qpinds = ols_low_high(op, en, moms, tol, max_mom=max_mom - 1.0)
    qpinds, ols_lower, ols_higher
end

function ps(Hdata, en, moms, tol::Float64=0.01; max_mom::Float64=Inf)
    H1,H2 = Hdata[1:2]
    
    ols_lower1, ols_higher1, inds1 = ols_low_high((H1 + H1')/2, en, moms, tol, max_mom=max_mom - 2.0)
    ols_lower2, ols_higher2, inds2 = ols_low_high((H2 + H2')/2, en, moms, tol, max_mom=max_mom - 2.0)
    pinds = intersect(inds1, inds2)
    ols_lower = ols_lower1 + ols_lower2
    ols_higher = ols_higher1 + ols_higher2
    
    pinds, ols_lower, ols_higher
end

function plot_qps(edata, Hdata, N::Int, tol::Float64=0.01; mom_res::Int=1, norm_fac=1.0, E0=nothing, max_mom::Float64=Inf, c::String="r", mrk="o")
    en, Tev = edata[1:2]
    E0 = (E0 == nothing ? en[1] : E0)
    en = norm_fac * (en - E0)
    
    spins = moms_Tev(Tev, N, mom_res)
    
    qpinds, ols = qps(Hdata, en, spins, tol, max_mom=max_mom)[1:2]
    
    plot(spins, en, "kx", markeredgewidth=0.7)
    scatter(spins, en, s=Float64[j in qpinds ? 100.0 : 0.0 for j in 1:length(ols)], alpha=0.8, edgecolors="None", color=c, zorder=10, marker=mrk)
    xlabel(L"s")
    ylabel(L"\Delta")
    
    qpinds, ols
end

function plot_ps(edata, Hdata, N::Int, tol::Float64=0.01; mom_res::Int=1, norm_fac=1.0, E0=nothing, max_mom::Float64=Inf, min_en::Float64=0.0, max_en::Float64=Inf, c::String="r", mrk="o", mark_pris::Bool=true)
    en, Tev = edata[1:2]
    E0 = (E0 == nothing ? en[1] : E0)
    en = norm_fac * (en - E0)
    
    spins = moms_Tev(Tev, N, mom_res)
    
    pinds, ols = ps(Hdata, en, spins, tol, max_mom=max_mom)[1:2]
    
    ind1 = findfirst(e-> e > min_en - sqrt(eps()), en)
    ind2 = findlast(e-> e < max_en + sqrt(eps()), en)
    plot(spins[ind1:ind2], en[ind1:ind2], "kx", markeredgewidth=0.7)
    if mark_pris
        scatter(spins[ind1:ind2], en[ind1:ind2], s=Float64[j in pinds ? 100.0 : 0.0 for j in ind1:ind2], alpha=0.8, edgecolors="None", color=c, zorder=10, marker=mrk)
    end
    xlabel(L"s")
    ylabel(L"\Delta")
    
    pinds = pinds[(pinds .>= ind1) & (pinds .<= ind2)]
    pinds, ols[ind1:ind2]
end

function plot_scaling(ev_data, Hs_data, normfacs, f, tol::Float64=0.01; Ns=sort(collect(keys(Hs_data))), plot_ratio::Bool=false, mom_res::Int=1, max_mom::Float64=Inf, min_err::Float64=0.0, min_en::Float64=0.0, max_en::Float64=Inf, fmt::String="o", Npow::Float64=1.0, fit::Bool=true, fit_Ns=Ns, col=nothing)
    data = Dict{Int, Any}()
    maxplot = 0
    for N in Ns
        en, Tev = ev_data[N][1:2]
        en -= en[1]
        en *= normfacs[N]
        spins = moms_Tev(Tev, N, mom_res)
        inds, ols_lower, ols_higher = f(Hs_data[N], en, spins, tol, max_mom=max_mom)
        ols_lower = ols_lower[inds]
        ols_higher = ols_higher[inds]
        flt = (abs(ols_lower) .> min_err) & (spins[inds] .> -sqrt(eps())) & (en[inds] .> min_en-sqrt(eps())) & (en[inds] .< max_en+sqrt(eps()))
        data[N] = (ols_lower[flt], ols_higher[flt], en[inds][flt], spins[inds][flt])
        maxplot = max(maxplot, countnz(flt))
    end
    
    xs = 1.0 ./ collect(Ns).^Npow
    
    res = []
    if plot_ratio
        for j in 1:maxplot
            ys = Float64[j <= length(data[N][1]) ? data[N][2][j] / data[N][1][j] : NaN for N in Ns]
            !all(isnan(ys)) && plot(xs, ys, fmt)
        end
        ylabel("Ratio of higher to lower energy overlaps")
    else
        for j in 1:maxplot
            yslo = Float64[j <= length(data[N][1]) ? data[N][1][j] : NaN for N in Ns]
            en = Float64[j <= length(data[N][1]) ? data[N][3][j] : NaN for N in Ns][end]
            mom = Float64[j <= length(data[N][1]) ? data[N][4][j] : NaN for N in Ns][end]
            if !all(isnan(yslo)) 
                push!(res, (xs, yslo, en, mom))
                
                if fit
                    flt = !isnan(yslo)
                    flt_Ns = Bool[N in fit_Ns for N in Ns[flt]]
                    a, b = linreg(xs[flt][flt_Ns], yslo[flt][flt_Ns])
                    @show a, b
                    fitx = linspace(0, maximum(xs))
                    plot(fitx, a + fitx * b, "-", color=col)
                end

                if !isnan(en + mom)
                    lbl = "\$ E\\approx $(round(real(en),1)), \\;p= $(round(Int, real(mom))) \$"
                else
                    lbl = ""
                end
                plot(xs, yslo, fmt, label=lbl, color=col)
            end
        end
        ylabel("Norm of overlap vector")
    end
    
    xlabel("\$1/N^{$(Npow)}\$")
    
    res
end

function towers(towerf, Hdata, pris::Vector{Int}, threshold::Float64=0.0)
    ols = zeros(Complex128, (size(Hdata[1],1), length(pris)))
    
    for j in 1:length(pris)
        ols[:,j] = towerf(Hdata, pris[j])
    end
    
    towers = zeros(Int, size(ols,1))
    olsabs = abs(ols)
    for j in 1:size(ols,1)
        towers[j] = indmax(olsabs[j,:])
        if olsabs[j, towers[j]] < threshold
            towers[j] = 0
        end
    end
    
    towers, ols
end

function H1_tower(Hdata, ev_num::Int)
    H1 = Hdata[1]
    
    op = expm(1.0im * (H1+H1'))
    ols = op[:,ev_num]
    ols[ev_num] = 0.0
    
    ols
end

function plot_H1_towers(edata, Hdata, norm_fac, N::Number, qpris::Vector{Int}; E0=nothing, cs=nothing, cdef=nothing, mom_res::Int=1, sz::Int=20, szp::Int=40, pex_yrange=0.12, pex_xspace=0.12, pex_rounddigits=6, maxDelta::Float64=Inf, tower_tol::Float64=1e-6, qpmarker="D", qpris_markflt=trues(length(qpris)))
    en, Tev = edata[1:2]
    E0 = (E0 == nothing ? en[1] : E0)
    en = norm_fac * (en - E0)
    spins = moms_Tev(Tev, N, mom_res)
    
    tows, ols = towers(H1_tower, Hdata, qpris, tower_tol)
    if cs == nothing
        cs_rest = nothing
    else
        cs_mod = [cdef, cs...]
        cs_rest = [cs_mod[ind+1] for ind in tows]
    end
    
    xs_p, ys_p = PlotTools.pointexclusion1D(spins[qpris], en[qpris], pex_yrange, pex_xspace, pex_rounddigits)
    flt = ys_p[qpris_markflt] .< maxDelta
    cs_flt = cs == nothing ? nothing : cs[qpris_markflt][flt]
    
    scatter(xs_p[qpris_markflt][flt], ys_p[qpris_markflt][flt], edgecolors=cs_flt, marker=qpmarker, s=szp, zorder=20, facecolors="none")
    
    flt = ys_p[!qpris_markflt] .< maxDelta
    cs_flt = cs == nothing ? nothing : cs[!qpris_markflt][flt]
    scatter(xs_p[!qpris_markflt][flt], ys_p[!qpris_markflt][flt], c=cs_flt, marker="o", s=sz, zorder=10)
    
    notqpris = trues(en)
    notqpris[qpris] = false
    xs, ys = PlotTools.pointexclusion1D(spins[notqpris], en[notqpris], pex_yrange, pex_xspace, pex_rounddigits)
    flt = ys .< maxDelta
    cs_rest_flt = cs_rest == nothing ? nothing : cs_rest[notqpris][flt]
    scatter(xs[flt], ys[flt], c=cs_rest_flt, marker="o", s=sz, zorder=10)
end

function H1H2_tower(Hdata, ev_num::Int)
    H1, H2 = Hdata[1:2]
    
    op = expm(1.0im * (H1+H1'+H2+H2'))
    ols = op[:,ev_num]
    ols[ev_num] = 0.0
    
    ols
end

function plot_H1H2_towers(edata, Hdata, norm_fac, N::Number, pris::Vector{Int}; E0=nothing, cs=nothing, mom_res::Int=1, sz::Int=20, szp::Int=40, pex_yrange=0.12, pex_xspace=0.12, pex_rounddigits=6, maxDelta::Float64=Inf)
    en, Tev = edata[1:2]
    E0 = (E0 == nothing ? en[1] : E0)
    en = norm_fac * (en - E0)
    spins = moms_Tev(Tev, N, mom_res)
    
    tows, ols = towers(H1H2_tower, Hdata, pris)
    if cs == nothing
        cs_rest = nothing
    else
        cs_rest = [cs[ind] for ind in tows]
    end
    
    xs_p, ys_p = PlotTools.pointexclusion1D(spins[pris], en[pris], pex_yrange, pex_xspace, pex_rounddigits)
    flt = ys_p .< maxDelta
    cs_flt = cs == nothing ? nothing : cs[flt]
    scatter(xs_p[flt], ys_p[flt], c=cs_flt, marker="D", s=szp, zorder=10)
    
    notpris = trues(en)
    notpris[pris] = false
    xs, ys = PlotTools.pointexclusion1D(spins[notpris], en[notpris], pex_yrange, pex_xspace, pex_rounddigits)
    flt = ys .< maxDelta
    cs_rest_flt = cs_rest == nothing ? nothing : cs_rest[notpris][flt]
    scatter(xs[flt], ys[flt], c=cs_rest_flt, marker="o", s=sz, zorder=10) 
end

end