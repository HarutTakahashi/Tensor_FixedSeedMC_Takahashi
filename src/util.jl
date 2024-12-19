using Random
import Tensor_FixedSeedMC as TCIMC


function makeMP0fromMPO(tt::Vector{Array{ComplexF64,3}})
    d = length(tt)
    tts = Vector{Array{ComplexF64,4}}()
    #M = Vector{ITensor}(undef, d)
    for i in 1:2:d
        #@show i
        size_core1 = size(tt[i])
        size_core2 = size(tt[i+1])
        mat_core1 = reshape(permutedims(tt[i], (1, 2, 3)), (size_core1[1] * size_core1[2], size_core1[3]))
        mat_core2 = reshape(permutedims(tt[i+1], (1, 2, 3)), (size_core2[1], size_core2[2] * size_core2[3]))
        res1 = mat_core1 * mat_core2
        res1 = reshape(res1, (size_core1[1], size_core1[2], size_core2[2], size_core2[3]))
        res1 = permutedims(res1, (1, 3, 2, 4))
        push!(tts, res1)
    end
    return tts
end


bonddims(M) = vcat([size(tt)[1] for tt in M], 1)


function generate_random_combinations(len, d, num_samples)
    Random.seed!(100)

    possible_values = collect(1:len)
    random_combinations = Vector{Int}[]

    for _ in 1:num_samples
        combination = rand(possible_values, d)
        push!(random_combinations, combination)
    end

    return random_combinations
end


function indices_1d_to_2d(indices, nx)
    # Flatten the result using a comprehension to extract elements of the tuple
    return vcat([x for idx in indices for x in index_1d_to_2d(idx, nx)]...)
end

function index_1d_to_2d(idx, nx)
    y = div(idx - 1, nx) + 1  # y座標（行番号）
    x = idx - (y - 1) * nx    # x座標（列番号）
    return (x, y)
end


function compComplexity(d, mps)
    chi = TCIMC.bonddims(mps)
    println("bond dimension for V: ", chi)

    complexity = 0
    for i in 1:d
        complexity += chi[i] * chi[i+1]
    end
    #complexity += chi[d-1]
    return complexity
end


function makeMP0fromMPS_2(tt::Vector{Array{ComplexF64,3}})
    d = length(tt)
    tts = Vector{Array{ComplexF64,4}}()
    #M = Vector{ITensor}(undef, d)
    for i in 1:3:d
        #@show i
        size_core1 = size(tt[i])
        size_core2 = size(tt[i+1])
        size_core3 = size(tt[i+2])
        mat_core1 = reshape(permutedims(tt[i], (1, 2, 3)), (size_core1[1] * size_core1[2], size_core1[3]))
        mat_core2 = reshape(permutedims(tt[i+1], (1, 2, 3)), (size_core2[1], size_core2[2] * size_core2[3]))
        new_core = mat_core1 * mat_core2
        new_core = reshape(new_core, (size_core1[1], size_core1[2] * size_core2[2], size_core2[3]))

        size_core_new = size(new_core)
        mat_core3 = reshape(permutedims(new_core, (1, 2, 3)), (size_core_new[1] * size_core_new[2], size_core_new[3]))
        mat_core4 = reshape(permutedims(tt[i+2], (1, 2, 3)), (size_core3[1], size_core3[2] * size_core3[3]))
        new_core2 = mat_core3 * mat_core4
        new_core2 = reshape(new_core2, size_core1[1], size_core1[2], size_core2[2] * size_core3[2], size_core3[3])
        res = permutedims(new_core2, (1, 3, 2, 4))
        push!(tts, res)
    end
    return tts
end



function reshape_MPStoMPO(tt, len1, len2)
    linkdims = TCIMC.bonddims(tt)
    new_tt = map(1:length(tt)) do i
        reshape(tt[i], (linkdims[i], len1, len2, linkdims[i+1]))
    end
    return new_tt
end