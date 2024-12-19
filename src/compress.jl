using LinearAlgebra
function compress!(tt::Vector{Array{T,3}}; tolerance=1e-12, maxbonddim=typemax(Int)) where {T}
    #tt_ = Vector{Array{Float64, 3}}[]
    #tt = copy(tt)
    for ell in 1:length(tt)-1
        shapel = size(tt[ell])
        left, right, newbonddim = _factorize_left(reshape(tt[ell], prod(shapel[1:2]), shapel[3]); tolerance=nothing, maxbonddim=nothing)
        tt[ell] = reshape(left, shapel[1:2]..., newbonddim)
        shaper = size(tt[ell+1])
        nexttensor = right * reshape(tt[ell+1], shaper[1], prod(shaper[2:3]))
        tt[ell+1] = reshape(nexttensor, newbonddim, shaper[2:3]...)
    end

    for ell in length(tt):-1:2
        shaper = size(tt[ell])
        left, right, newbonddim = _factorize_right(reshape(tt[ell], shaper[1], prod(shaper[2:3])); tolerance=tolerance, maxbonddim=maxbonddim)
        tt[ell] = reshape(right, newbonddim, shaper[2:3]...)
        shapel = size(tt[ell-1])
        nexttensor = reshape(tt[ell-1], prod(shapel[1:2]), shapel[3]) * left
        tt[ell-1] = reshape(nexttensor, shapel[1:2]..., newbonddim)
    end
    return nothing
end

function _factorize_left(A::Matrix{V}; tolerance::Union{Float64,Nothing}=nothing, maxbonddim::Union{Int,Nothing}=nothing)::Tuple{Matrix{V},Matrix{V},Int} where {V}
    factorization = svd(A)
    if isnothing(tolerance) || isnothing(maxbonddim)
        # No truncation
        notrunc = length(factorization.S)
        return (
            factorization.U[:, :], Diagonal(factorization.S[:]) * factorization.Vt[:, :], notrunc
        )
    else
        # With truncation
        trunci = min(replacenothing(findlast(>(tolerance), factorization.S / maximum(factorization.S)), 1), maxbonddim)
        return (
            factorization.U[:, 1:trunci], Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :], trunci
        )
    end
end

function _factorize_right(A::Matrix{V}; tolerance::Union{Float64,Nothing}=nothing, maxbonddim::Union{Int,Nothing}=nothing)::Tuple{Matrix{V},Matrix{V},Int} where {V}
    factorization = svd(A)
    if isnothing(tolerance) || isnothing(maxbonddim)
        # No truncation
        notrunc = length(factorization.S)
        return (
            factorization.U[:, :] * Diagonal(factorization.S[:]), factorization.Vt[:, :], notrunc
        )
    else
        # With truncation
        trunci = min(replacenothing(findlast(>(tolerance), factorization.S / maximum(factorization.S)), 1), maxbonddim)
        return (
            factorization.U[:, 1:trunci] * Diagonal(factorization.S[1:trunci]), factorization.Vt[1:trunci, :], trunci
        )
    end
end

function replacenothing(value::Union{T,Nothing}, default::T)::T where {T}
    if isnothing(value)
        return default
    else
        return value
    end
end