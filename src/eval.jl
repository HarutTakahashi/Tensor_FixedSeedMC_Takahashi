using Random
import Tensor_FixedSeedMC as TCIMC

function evaluate_options(random_combinations, tt_option, η, d, r, T, t)
    result_hako = []
    result_time = []
    for i in random_combinations
        #@show i
        time_inner = @elapsed begin
            inner_ = tt_evaluate(tt_option, i)
            result = ((η^d) * exp(-r * (T - t)) * inner_) / ((2 * π)^d)
            result = real(result)
        end
        push!(result_hako, result)
        push!(result_time, time_inner)
    end
    return result_hako, result_time
end

function tt_evaluate(tt::Vector{Array{V,3}}, indexset) where {V}
    only(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
end

function _evaluate(tt::Vector{Array{V,3}}, indexset) where {V}
    only(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
end


function _evaluate3(tt::Vector{Array{V,4}}, indexset) where {V}
    only(prod(T[:, i, j, :] for (T, (i, j)) in zip(tt, indexset)))
end

function evaluate_options2(random_combinations, tt_option, η, d, r, T, t)
    result_hako = []
    result_time = []
    for i in random_combinations
        indices = TCIMC.indices_1d_to_2d(i, 100)
        indices_pairs = [(indices[i], indices[i+1]) for i in 1:2:length(indices)-1]
        #@show i
        time_inner = @elapsed begin
            inner_ = _evaluate3(tt_option, indices_pairs)
            result = ((η^d) * exp(-r * (T - t)) * inner_) / ((2 * π)^d)
            result = real(result)
        end
        push!(result_hako, result)
        push!(result_time, time_inner)
    end
    return result_hako, result_time
end
