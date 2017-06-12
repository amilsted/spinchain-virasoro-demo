module CFTutils

using PyPlot
using PyCall
using PlotTools

immutable CFTField
    w::Float64 #conformal weight
    d::Vector{Int} #Virasoro characters
end

immutable CFTOp
    fh::CFTField
    fa::CFTField
    lbl::AbstractString
end

sdim(oph::CFTField, Ih::Int, opa::CFTField, Ia::Int) = (oph.d[Ih+1] * opa.d[Ia+1], oph.w + Ih + opa.w + Ia, oph.w + Ih - opa.w - Ia)

sdim(op::CFTOp, Ih::Int, Ia::Int) = sdim(op.fh, Ih, op.fa, Ia)

function sop_count(oph::CFTField, opa::CFTField, maxsd::Float64, maxspin::Float64=Inf)
    Imax = floor(Int, maxsd - (oph.w + opa.w))
    numops = 0
    for I in 0:Imax
        for Ih in 0:I
            d, sd, sp = sdim(oph, Ih, opa, I-Ih)
            if abs(sp) <= maxspin
                numops += d
            end
        end
    end
    numops
end

sop_count(op::CFTOp, maxsd::Float64=15.0, maxspin::Float64=Inf) = sop_count(op.fh, op.fa, maxsd, maxspin)

function sops(oph::CFTField, opa::CFTField, maxsd::Float64=15.0)
    Imax = floor(Int, maxsd - (oph.w + opa.w))
    
    sds = Float64[]
    sps = Float64[]
    ds = Int[]
    for I in 0:Imax
        for Ih in 0:I
            d, sd, sp = sdim(oph, Ih, opa, I-Ih)
            push!(ds, d)
            push!(sds, sd)
            push!(sps, sp)
        end
    end
    sds, sps, ds
end

sops(op::CFTOp, maxsd::Float64=15.0) = sops(op.fh, op.fa, maxsd)

function sops_by_s(op::CFTOp, maxsd::Float64=15.0; maxspin::Float64=Inf)
    sds, sps, ds = sops(op, maxsd)
    
    ss = unique(round(sps,6))
    ss = ss[abs(ss) .< maxspin + sqrt(eps())]
    
    haspri = Bool[]
    ress = Vector{Float64}[]
    for s in ss
        flt = abs(sps - s) .< 1e-6
        
        push!(haspri, flt[1])
        
        sds_s = sds[flt]
        sps_s = sps[flt]
        ds_s = ds[flt]
        
        res = Float64[]
        for j in 1:length(sds_s)
            for k in 1:ds_s[j]
                append!(res, sds_s[j])
            end
        end
        push!(ress, res)
    end
    ss, ress, haspri
end

function plot_sops_v3(ops::Vector{CFTOp}, cs::Vector, maxsd::Float64=15.0; minsd::Float64=-Inf, maxspin::Float64=Inf, xsep=0.12, ysep=0.12, szp=40, sz=20, shift_pris::Bool=true)    
    sops = [sops_by_s(op, maxsd, maxspin=maxspin) for op in ops]
    
    ss = unique(round(vcat([sop[1] for sop in sops]...),6))
    sds = Vector{Float64}[]
    cols = Vector[]
    ispri = Vector{Bool}[]
    for s in ss
        sds_s = Float64[]
        cols_s = []
        ispri_s = []
        for (j, (ss_op, sd_op, haspri_op)) in enumerate(sops)
            ind = findfirst(s_op->s_op == s, ss_op)
            if ind > 0 
                sd_op = sd_op[ind]
                if length(sd_op) > 0
                    append!(sds_s, sd_op)
                    append!(cols_s, [cs[j] for k in 1:length(sd_op)])
                    is_pri_op = falses(length(sd_op))
                    is_pri_op[1] = haspri_op[ind]
                    append!(ispri_s, is_pri_op)
                end
            end
        end
        srt = sortperm(sds_s)
        sds_s = sds_s[srt]
        cols_s = cols_s[srt]
        ispri_s = ispri_s[srt]
        
        push!(sds, sds_s)
        push!(cols, cols_s)
        push!(ispri, ispri_s)
    end
    
    if shift_pris
        for j in 1:length(ss)
            xs = PlotTools.pointexclusion1D(sds[j], ysep, xsep)
            xs += ss[j]
            
            sdflt = sds[j] .> minsd
            if any(ispri[j][sdflt])
                xs_p = xs[sdflt][ispri[j][sdflt]]
                sds_p = sds[j][sdflt][ispri[j][sdflt]]
                cols_p = cols[j][sdflt][ispri[j][sdflt]]
                scatter(xs_p, sds_p, c=cols_p, marker="D", s=szp, zorder=10)
            end
            
            scatter(xs[sdflt][!ispri[j][sdflt]], sds[j][sdflt][!ispri[j][sdflt]], c=cols[j][sdflt][!ispri[j][sdflt]], marker="o", s=sz, zorder=10)
        end
    else
        for j in 1:length(ss)
            sdflt = sds[j] .> minsd
            
            if any(ispri[j][sdflt])
                xs = PlotTools.pointexclusion1D(sds[j][sdflt][ispri[j][sdflt]], ysep, xsep) + ss[j]
                scatter(xs, sds[j][sdflt][ispri[j][sdflt]], c=cols[j][sdflt][ispri[j][sdflt]], marker="D", s=szp, zorder=10)
            end
        
            xs = PlotTools.pointexclusion1D(sds[j][sdflt][!ispri[j][sdflt]], ysep, xsep)
            xs += ss[j]
            scatter(xs, sds[j][sdflt][!ispri[j][sdflt]], c=cols[j][sdflt][!ispri[j][sdflt]], marker="o", s=sz, zorder=10)
        end
    end

end

const ising_cft_1_1 = CFTField(0.0, Int[1, 0, 1, 1, 2, 2, 3, 3, 5, 5, 7, 8, 11, 12, 16, 18])
const ising_cft_1_2 = CFTField(1/16, Int[1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27])
const ising_cft_1_3 = CFTField(1/2, Int[1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 9, 12, 14, 17, 20])

const ising_cft_eye = CFTOp(ising_cft_1_1, ising_cft_1_1, "\\mathbb{I}")
const ising_cft_sigma = CFTOp(ising_cft_1_2, ising_cft_1_2, "\\sigma")
const ising_cft_epsilon = CFTOp(ising_cft_1_3, ising_cft_1_3, "\\epsilon")

const potts3_cft_1_1 = CFTField(0.0, Int[1, 0, 1, 1, 2, 2, 4, 4, 7, 8, 12, 14, 21, 24, 34, 41])
const potts3_cft_1_2 = CFTField(1/8, Int[1, 1, 1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 35, 45, 58, 75])
const potts3_cft_1_3 = CFTField(2/3, Int[1, 1, 2, 2, 4, 5, 8, 10, 15, 19, 27, 34, 46, 58, 77, 96])
const potts3_cft_1_4 = CFTField(13/8, Int[1, 1, 2, 3, 4, 6, 9, 12, 16, 22, 29, 38, 50, 64, 82, 105])
const potts3_cft_2_1 = CFTField(2/5, Int[1, 1, 1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 35, 45, 58, 74])
const potts3_cft_2_3 = CFTField(1/15, Int[1, 1, 2, 3, 5, 7, 10, 14, 20, 26, 36, 47, 63, 81, 106, 135])
const potts3_cft_3_1 = CFTField(7/5, Int[1, 1, 2, 2, 4, 5, 8, 10, 15, 19, 26, 33, 45, 56, 74, 92])
const potts3_cft_4_1 = CFTField(3.0, Int[1, 1, 2, 3, 4, 5, 8, 10, 14, 18, 24, 31, 41, 51, 66, 83])

const potts3_cft_eye = CFTOp(potts3_cft_1_1, potts3_cft_1_1, "\\mathbb{I}")
const potts3_cft_sigma = CFTOp(potts3_cft_2_3, potts3_cft_2_3, "\\sigma")
const potts3_cft_epsilon = CFTOp(potts3_cft_2_1, potts3_cft_2_1, "\\epsilon")
const potts3_cft_Z = CFTOp(potts3_cft_1_3, potts3_cft_1_3, "Z")
const potts3_cft_X_eps = CFTOp(potts3_cft_3_1, potts3_cft_2_1, "(X,\\epsilon)")
const potts3_cft_eps_X = CFTOp(potts3_cft_2_1, potts3_cft_3_1, "(\\epsilon,X)")
const potts3_cft_X = CFTOp(potts3_cft_3_1, potts3_cft_3_1, "X")
const potts3_cft_Y = CFTOp(potts3_cft_4_1, potts3_cft_4_1, "Y")
const potts3_cft_Y_eye = CFTOp(potts3_cft_4_1, potts3_cft_1_1, "(Y,\\mathbb{I})")
const potts3_cft_eye_Y = CFTOp(potts3_cft_1_1, potts3_cft_4_1, "(\\mathbb{I},Y)")

end