module EDutils

using Iterators
using LinearMaps
using TensorOperations

"""
Applies operator `op` to state `s` at sites `n` to `n+r-1`, 
where `r` is the number of sites on which `op` acts nontrivially (the range).

Both `s` and `op` must be tensors with one index per site:

A state on `Ns` sites has `ndim(s) == Ns` .

An operator taking `Ns_op` sites to `Ns_op_out` sites has `ndim(op) == Ns_op_out + Ns_op`.
The `Ns_op` indices acting on the state are assumed to be the last `Ns_op` indices.

Periodic boundary conditions are used to act on sites past the end of the chain.
"""
function apply_op_at!(s, res, op, n::Int; α=1.0, β=0.0, Ns_op_out=0, s_sz=size(s))
    Ns = length(s_sz)
    op_sz = size(op)
    
    if Ns_op_out == 0
        Ns_op = ndims(op) ÷ 2
        Ns_op_out = Ns_op
    else
        Ns_op = ndims(op) - Ns_op_out
    end
    
    if n == 1 
        #Make this more obviously just matrix multiplication where possible. Why does tensorcontract!() fail to do this by itself?
        s = reshape(s, (prod(s_sz[1:Ns_op]), prod(s_sz[Ns_op+1:end])))
        op = reshape(op, (prod(op_sz[1:Ns_op_out]), size(s, 1)))
        res_r = reshape(res, (size(op,1), size(s, 2)))
        tensorcontract!(α, op, [:o1, :o2], 'N', s, [:o2, :sR], 'N', β, res_r, [:o1, :sR])
    else
        #Must avoid crossing the max. tuple-length limit for type-inference.
        #This is the general case. Some of these dimensions will be of size 1.
        
        s = reshape(s, (prod(s_sz[1:n + Ns_op - 1 - Ns]), #wrapped operator index
                        prod(s_sz[max(n + Ns_op - Ns, 1):n - 1]), #rest left
                        prod(s_sz[n:min(n + Ns_op - 1, Ns)]), #operator index
                        prod(s_sz[n + Ns_op:Ns]))) #rest right
        
        op_wrap_ind = min(Ns_op_out, Ns - n + 1)
        op = reshape(op, (prod(op_sz[1:op_wrap_ind]), #out
                          prod(op_sz[op_wrap_ind+1:Ns_op_out]), #out, wrapped
                          size(s, 3), size(s, 1))) #in

        res_r = reshape(res, (size(op, 2), size(s, 2), size(op, 1), size(s, 4))) #this should not alter the storage

        opinds = [:op_o1, :op_o2, :op1, :op2]
        inds = [:op2, :sL, :op1, :sR]
        inds_out = [:op_o2, :sL, :op_o1, :sR]

        tensorcontract!(α, op, opinds, 'N', s, inds, 'N', β, res_r, inds_out)
    end
    
    res
end

"""
Translates a state by `n` sites (site 1 becomes site N, site 2 becomes site 1, and so on).
"""
function translate_pbc!(s, st, n::Int, a::Number, b::Number, s_sz)
    Ns = length(s_sz)
    st_sz = size(st)
    
    #ensure type-inference works by reducing to a minimal number of indicies.
    if n >= 0
        s = reshape(s, (prod(s_sz[1:n]), prod(s_sz[n+1:end])))
    else
        s = reshape(s, (prod(s_sz[1:Ns+n]), prod(s_sz[Ns+n+1:end])))
    end
    
    st = reshape(st, (size(s,2), size(s,1)))
    
    if a == 1.0 && b == 0.0
        transpose!(st, s)
    else
        tensoradd!(a, s, (:s1, :s2), b, st, (:s2, :s1)) #at least in Julia 0.5, tensoradd!() is slower, particularly for n < 0
    end
    
    st = reshape(st, st_sz)
    
    st, ntuple(j->s_sz[mod1(j+n,Ns)], Ns)
end

function translate_pbc!(s, st, n::Int; a=1.0, b=0.0, s_sz=size(s), ret_size::Bool=false)
    st, s_sz = translate_pbc!(s, st, n, a, b, s_sz)
    ret_size ? (st, s_sz) : st
end

function translate_pbc(s, n::Int; s_sz=size(s), ret_size::Bool=false)
    res = zeros(s)
    translate_pbc!(s, res, n, s_sz=s_sz, ret_size=ret_size)
end

function apply_global_prod_op!(s::AbstractArray, ops::Vector{AbstractArray}; tmp = zeros(s), s_sz=size(s))
    for n in 1:length(s_sz)
        apply_op_at!(s, tmp, ops[n], 1, s_sz=s_sz)
        s, s_sz = translate_pbc!(tmp, s, 1, 1.0, 0.0, s_sz)
    end
    s
end

function apply_global_prod_op!(s::AbstractArray, lop::AbstractArray; tmp = zeros(s), s_sz=size(s))
    apply_global_prod_op!(s, AbstractArray[lop for j in 1:length(s_sz)], tmp=tmp, s_sz=s_sz)
end

#NOTE: Modifies s!
function apply_local_op!{T}(s::Vector{T}, s_sz::Tuple{Vararg{Int}}, H::Vector{Array}, s_tmp::Vector{T}, res_tmp1::Vector{T}, res_tmp2::Vector{T})
    fill!(res_tmp1, 0.0)
    for n in 1:length(s_sz)
        apply_op_at!(s, res_tmp1, H[n], 1, β=1.0, s_sz=s_sz)

        translate_pbc!(res_tmp1, res_tmp2, 1, 1.0, 0.0, s_sz) #translating by +1 is usually faster than -1
        tmp = res_tmp1
        res_tmp1 = res_tmp2
        res_tmp2 = tmp

        if n < length(s_sz)
            translate_pbc!(s, s_tmp, 1, 1.0, 0.0, s_sz)
            tmp = s
            s = s_tmp
            s_tmp = tmp
        end
    end
    res_tmp1
end

function apply_local_op{T}(s::Vector{T}, s_sz::Tuple{Vararg{Int}}, H::Vector{Array})
    s_vlen = prod(s_sz)
    res1 = zeros(T, s_vlen)
    res2 = zeros(T, s_vlen)
    s1 = copy(s)
    s2 = zeros(T, s_vlen)
    apply_local_op!(s1, s_sz, H, s2, res1, res2)
end

function apply_local_op(s::Vector, s_sz::Tuple{Vararg{Int}}, Hloc::Array)
    Ns = length(s_sz)
    H = Array[Hloc for n in 1:Ns]
    apply_local_op(s, s_sz, H)
end

#The part of the Hamiltonian spectrum obtained is determined by the spectrum and the choice of "which" parameter.
#To get the low-energy states, ensure the spectrum of H is negative.
#Resolving momenta requires H to have real spectrum and be positive or negative definite.
function eigs_localHam_circle(H::Vector{Array}; momenta::Bool=false, mom_dir=:L, symm_func=nothing, nev::Int=41, ncv::Int=max(20, 2*nev+1), which=:LM)
    Ns = length(H)
    Nh = ndims(H[1]) ÷ 2
    D = size(H[1], 1)
    
    s_sz = ntuple(x->D, Ns)
    s_vlen = prod(s_sz)
    @show Ns, Nh, s_sz
    
    #We work with vectors because Julia freaks when the number of array dimensions gets too high (type-inference fails).
    res1 = zeros(Complex128, s_vlen)
    res2 = zeros(Complex128, s_vlen)
    s1 = zeros(Complex128, s_vlen)
    s2 = zeros(Complex128, s_vlen)
    s3 = zeros(Complex128, s_vlen)
    function f!(dest::AbstractVector{Complex128}, s_in::AbstractVector{Complex128})
        copy!(s1, s_in) 
        #s_in may be a subarray... make sure we have an Array, otherwise TensorOps complains. 
        #Also creating a reshapedsubarray can be slooooow.

        if momenta
            s2 = translate_pbc!(s1, s2, mom_dir == :L ? 1 : -1, s_sz=s_sz)
            tmp = s1
            s1 = s2
            s2 = tmp
        end
        
        if symm_func != nothing
            s1 = symm_func(s1, s2, s3, s_sz)
        end
        
        Hsr = apply_local_op!(s1, s_sz, H, s2, res1, res2)
        copy!(dest, Hsr)
        dest
    end
    
    #FIXME: Calling convention changed since 0.2.0
    Hop = LinearMap(f!, s_vlen, Complex128, ishermitian=false, ismutating=true)

    ev, eV, nconv, niter, nmult, resid = eigs(Hop, nev=nev, ritzvec=true, which=which, ncv=ncv)
    
    if momenta
        Hev = -abs(ev)
        Tev = ev ./ Hev
        return Hev, Tev, eV
    else
        return ev, eV
    end
end

function eigs_localHam_circle(Hloc::Array, Ns::Int; momenta::Bool=false, mom_dir=:L, symm_func=nothing, nev::Int=41, ncv::Int=max(20, 2*nev+1))
    
    Nh = ndims(Hloc) ÷ 2
    D = size(Hloc, 1)
    Hlocr = reshape(Hloc, (D^Nh, D^Nh))
    em = maximum(real(eigvals(Hlocr)))

    Hlocr = Hlocr - I * em #offset so that we can find eigenvalues with largest magnitude
    Hloc = reshape(Hlocr, size(Hloc))
    
    hs = Array[Hloc for n in 1:Ns]
    
    res = eigs_localHam_circle(hs, momenta=momenta, mom_dir=mom_dir, symm_func=symm_func, nev=nev)
    
    res = (res[1] + em * Ns, res[2:end]...)
    
    res
end

function get_Pmoms(ps::Vector{Float64})
    (s, tmp1, tmp2, s_sz)->begin
        copy!(tmp2, s)
        fill!(s, 0.0)
        
        N = length(s_sz)
        for n in 1:N
            tmp1 = translate_pbc!(tmp2, tmp1, 1, s_sz=s_sz)
            tmp1, tmp2 = (tmp2, tmp1)
        
            LinAlg.BLAS.axpy!(sum(cis(-n * p * 2pi / N) for p in ps), tmp2, s)
        end
        
        scale!(s, 1/N)
        s
    end
end

#-----------------------------

const pauliX = [0.0 1.0;
                1.0 0.0]
const pauliY = 1.0im*[0.0 -1.0;
                      1.0  0.0]
const pauliZ = [1.0  0.0;
                0.0 -1.0]

function ham_ising_comp(λ::Number=1.0, hz::Number=1.0, hx::Number=0.0)
    -(hz*pauliZ + hx*pauliX), -λ * kron(pauliX, pauliX)
end

function ham_ising(λ::Number=1.0, hz::Number=1.0, hx::Number=0.0)
    h1, h2 = ham_ising_comp(λ, hz, hx)
    
    h2 + 1/2 * kron(h1, eye(2)) + 1/2 * kron(eye(2), h1)
end

function weylops(p::Int)
    om = cis(2π / p)
    U = diagm(Complex128[om^j for j in 0:p-1])
    V = diagm(ones(p - 1), 1)
    V[end, 1] = 1
    U, V, om
end

function weylops_basis(p::Int)
    U, V, om = weylops(p)
    ops = Matrix[(U^j for j in 1:p-1)..., (V^j for j in 1:p-1)..., (U^j * V^k for j in 1:p-1 for k in 1:p-1)...]
    lbls = [("U^$j" for j in 1:p-1)..., ("V^$j" for j in 1:p-1)..., ("U^$(j)V^$(k)" for j in 1:p-1 for k in 1:p-1)...]
    ops, lbls
end

function ham_vpotts_comp(p::Int, J::Float64=1.0, h::Float64=1.0, hU::Number=0.0)
    U, V, om = weylops(p)
    
    UUh = kron(U, U')
    
    -0.5 * h * (V + V') - hU * 1/3 * (U + U' + I), -0.5 * J * (UUh + UUh')
end

function ham_vpotts(p::Int, J::Float64=1.0, h::Float64=1.0, hU::Number=0.0)
    h1, h2 = ham_vpotts_comp(p, J, h, hU)
    
    h2 + 1/2 * kron(h1, eye(p)) + 1/2 * kron(eye(p), h1)
end
                                
                                
#---------------------------------
                                
function ed(N::Int, h::Array, nev::Int; symm_func=nothing, ncv::Int=max(20, 2*nev+1))
    @time en_pbc, mom_pbc, eV_pbc = eigs_localHam_circle(h, N, momenta=true, symm_func=symm_func, nev=nev, ncv=ncv)
    return en_pbc, mom_pbc, eV_pbc
end

function ed!(ev_data, Ns::Vector{Int}, h::Array, nev::Int; symm_func=nothing)
    for N in Ns
        println("Starting $N")
        if !haskey(ev_data, N)
            ev_data[N] = ed(N, h, nev, symm_func=symm_func)
        end

        eV = ev_data[N][3]
        eVisU = vecnorm(eV' * eV - I)
        println("Is eV isometry? $(eVisU)")
    end
    ev_data
end

end