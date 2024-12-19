# 微分演算子を作成する関数
function diff_matrix(N, Δx)
    # Nはグリッド点の数, Δxはグリッド間隔
    D = zeros(Float64, N, N)  # N×Nのゼロ行列を作成

    # 境界条件の設定（前進差分と後退差分）
    D[1, 1] = -1 / Δx
    D[1, 2] = 1 / Δx
    D[N, N-1] = -1 / Δx
    D[N, N] = 1 / Δx

    # 内部点の設定（中心差分）
    for i in 2:N-1
        D[i, i-1] = -1 / (2 * Δx)
        D[i, i+1] = 1 / (2 * Δx)
    end

    return D
end

function second_diff_matrix(n, h)
    D2 = zeros(n, n)  # n x n のゼロ行列を作成

    # 前進差分（1行目）
    D2[1, 1] = 1
    D2[1, 2] = -2
    D2[1, 3] = 1

    # 中心差分（内部の点）
    for i in 2:n-1
        D2[i, i-1] = 1
        D2[i, i] = -2
        D2[i, i+1] = 1
    end

    # 後退差分（最後の行）
    D2[n, n-2] = 1
    D2[n, n-1] = -2
    D2[n, n] = 1

    # h^2 で割って微分スケールに調整
    return D2 / (h^2)
end


function grad_tt(tt::Vector{Array{T,3}}, ind::Int64, diff_mat::Matrix{Float64}) where {T}
    mat_size = size(diff_mat)
    size_core = size(tt[ind])
    if mat_size[2] != size_core[2]
        error("Error: mat_size[2] ($(mat_size[2])) does not match size_core[2] ($(size_core[2]))")
    end

    core = reshape(permutedims(tt[ind], (2, 1, 3)), size_core[2], size_core[1] * size_core[3])
    #grad_core = reshape(diff_mat * core, size_core[1], size_core[2], size_core[3])
    grad_core_ = reshape(diff_mat * core, size_core[2], size_core[1], size_core[3])
    grad_core = permutedims(grad_core_, (2, 1, 3))

    tt = map(1:length(tt), tt) do i, x
        i == ind ? grad_core : x
    end

    #=
    tt = map(1:length(tt)) do i
        if i == ind
            return grad_core
        else
            return tt[i]
        end
    end
    =#
    return tt
end

# N個のコアに対する全通りの二回微分を実行する関数
function second_order_diff(option_tt, N, diff_mat)
    new_tt_list = []  # 二回微分結果を格納するリスト

    for i in 1:N
        for j in 1:N
            # i 番目のコアに対する一回目の微分
            first_diff = grad_tt(option_tt, i, diff_mat)
            # j 番目のコアに対する二回目の微分
            second_diff = grad_tt(first_diff, j, diff_mat)

            # 結果をリストに保存
            push!(new_tt_list, second_diff)
        end
    end

    return new_tt_list  # 二回微分の全結果
end



function grad_tt_sigma(tt::Vector{Array{T,4}}, ind::Int64, diff_mat::Matrix{Float64}) where {T}
    mat_size = size(diff_mat)
    size_core = size(tt[ind])
    if mat_size[2] != size_core[2]
        error("Error: mat_size[2] ($(mat_size[2])) does not match size_core[2] ($(size_core[2]))")
    end

    core = reshape(permutedims(tt[ind], (2, 1, 3, 4)), size_core[2], size_core[1] * size_core[3] * size_core[4])
    grad_core = reshape(diff_mat * core, size_core[2], size_core[1], size_core[3], size_core[4])
    grad_core = permutedims(grad_core, (2, 1, 3, 4))

    tt = map(1:length(tt)) do i
        if i == ind
            return grad_core
        else
            return tt[i]
        end
    end
    return tt
end

function grad_tt_stock0(tt::Vector{Array{T,4}}, ind::Int64, diff_mat::Matrix{Float64}) where {T}
    mat_size = size(diff_mat)
    size_core = size(tt[ind])
    if mat_size[2] != size_core[3]
        error("Error: mat_size[2] ($(mat_size[2])) does not match size_core[2] ($(size_core[2]))")
    end

    core = reshape(permutedims(tt[ind], (3, 1, 2, 4)), size_core[3], size_core[1] * size_core[2] * size_core[4])
    grad_core = reshape(diff_mat * core, size_core[3], size_core[1], size_core[2], size_core[4])
    grad_core = permutedims(grad_core, (2, 3, 1, 4))

    tt = map(1:length(tt)) do i
        if i == ind
            return grad_core
        else
            return tt[i]
        end
    end
    return tt
end