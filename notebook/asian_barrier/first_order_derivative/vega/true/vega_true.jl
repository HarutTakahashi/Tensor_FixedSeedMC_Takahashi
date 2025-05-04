##########################################################
# パッケージの読み込みとエイリアスの設定
##########################################################
using Revise, LinearAlgebra, LaTeXStrings, Plots, CSV, DataFrames, JLD2, Random, Statistics, Distributions
using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
using Tensor_FixedSeedMC
import Tensor_FixedSeedMC as TCIMC
using Statistics, Random, Distributions

# アジアン・バリアオプションのモンテカルロシミュレーション（インデックス渡し版）
function asian_barrier_option_tci(j, T, K, B, S0s, sigmas, mu, r, d_normals, N_STEPS, N_PATHS)
    tmp1 = mu * T / N_STEPS
    tmp2 = exp(-r * T)
    tmp3 = sqrt(T / N_STEPS)
    discounted_payoffs = zeros(Float64, N_PATHS)
    S0 = S0s[j[1]]
    sigma = sigmas[j[2]]
    for i in 1:N_PATHS
        s_curr = S0
        running_avg = 0.0
        for n in 1:N_STEPS
            s_curr += tmp1 * s_curr + sigma * s_curr * tmp3 * d_normals[i + (n-1)*N_PATHS]
            running_avg += (s_curr - running_avg) / n
            if running_avg <= B
                break
            end
        end
        discounted_payoffs[i] = tmp2 * max(running_avg - K, 0)
    end
    mean_payoff = mean(discounted_payoffs)
    std_payoff = std(discounted_payoffs)
    error = std_payoff / sqrt(N_PATHS)
    return mean_payoff
end

# アジアン・バリアオプションのモンテカルロシミュレーション（インデックス渡し版）
function asian_barrier_option_error(j, T, K, B, S0s, sigmas, mu, r, d_normals, N_STEPS, N_PATHS)
    tmp1 = mu * T / N_STEPS
    tmp2 = exp(-r * T)
    tmp3 = sqrt(T / N_STEPS)
    discounted_payoffs = zeros(Float64, N_PATHS)
    S0 = S0s[j[1]]
    sigma = sigmas[j[2]]
    for i in 1:N_PATHS
        s_curr = S0
        running_avg = 0.0
        for n in 1:N_STEPS
            s_curr += tmp1 * s_curr + sigma * s_curr * tmp3 * d_normals[i + (n-1)*N_PATHS]
            running_avg += (s_curr - running_avg) / n
            if running_avg <= B
                break
            end
        end
        discounted_payoffs[i] = tmp2 * max(running_avg - K, 0)
    end
    mean_payoff = mean(discounted_payoffs)
    std_payoff = std(discounted_payoffs)
    error = std_payoff / sqrt(N_PATHS)
    return mean_payoff, error
end

# TCIワンショット補間関数
function tci_oneshot(func, d, localdims, firstpivot, tol)
    BLAS.set_num_threads(4)
    for _ in 1:100
        p = TCI.optfirstpivot(func, localdims, firstpivot)
        if abs(func(p)) > abs(func(firstpivot))
            firstpivot = p
        end
    end
    qtt, ranks, errors = TCI.crossinterpolate2(Float64, func, localdims, [firstpivot],
                                                  tolerance=tol, maxiter=6, verbosity=1,
                                                  loginterval=1, pivotsearch=:rook)
    return qtt, errors
end

#―――――――――――――――――――――――――――――――――――――――
# 1) パスごとの割引後ペイオフを返す関数
#―――――――――――――――――――――――――――――――――――――――
function asian_barrier_payoffs(
    j, T, K, B,
    S0s, sigmas, mu, r,
    d_normals, N_STEPS, N_PATHS
) :: Vector{Float64}
    tmp1 = mu * T / N_STEPS
    tmp2 = exp(-r * T)
    tmp3 = sqrt(T / N_STEPS)

    payoffs = zeros(Float64, N_PATHS)
    S0 = S0s[j[1]]
    sigma = sigmas[j[2]]

    for i in 1:N_PATHS
        s = S0
        avg = 0.0
        for n in 1:N_STEPS
            s += tmp1 * s + sigma * s * tmp3 * d_normals[i + (n-1)*N_PATHS]
            avg += (s - avg) / n
            if avg <= B
                # バリア割れならペイオフ0でループ脱出
                avg = 0.0
                break
            end
        end
        payoffs[i] = tmp2 * max(avg - K, 0)
    end
    return payoffs
end

"""
    vega_asian_barrier(
        j, T, K, B,
        S0s, sigmas, mu, r,
        d_normals, N_STEPS, N_PATHS;
        h_sigma = 1e-4
    ) -> (Float64, Float64)

アジアンバリアオプションの Vega を中心差分で近似し、(推定値, 推定誤差) を返す関数。

- j: (iS0, iSigma)
- h_sigma: ボラティリティに対する小さな変化量
"""
function vega_asian_barrier(
    j, T, K, B,
    S0s, sigmas, mu, r,
    d_normals, N_STEPS, N_PATHS;
    h = 1e-4
) :: Tuple{Float64, Float64}

    # 1) ボラティリティを ±h_sigma ずらしたベクトルを作成
    sigmas_up   = copy(sigmas)
    sigmas_down = copy(sigmas)

    sigmas_up[j[2]]   = sigmas[j[2]] + h
    sigmas_down[j[2]] = sigmas[j[2]] - h

    # 2) 上側 (sigma + h_sigma) のペイオフ
    pay_plus = asian_barrier_payoffs(
        j, T, K, B,
        S0s, sigmas_up, mu, r,
        d_normals, N_STEPS, N_PATHS
    )

    # 3) 下側 (sigma - h_sigma) のペイオフ
    pay_minus = asian_barrier_payoffs(
        j, T, K, B,
        S0s, sigmas_down, mu, r,
        d_normals, N_STEPS, N_PATHS
    )

    # 4) パスごとの価格差分 (centered difference)
    #    Vega = [C(σ + h_sigma) - C(σ - h_sigma)] / (2*h_sigma)
    vega_path = (pay_plus .- pay_minus) ./ (2*h)

    # 5) 平均と標準誤差
    vega_mean = mean(vega_path)
    vega_err  = std(vega_path) / sqrt(N_PATHS)

    return vega_mean, vega_err
end


##########################################################
# パラメータの設定とグリッド生成
##########################################################
d = 2
N_STEPS = 365
N_PATHS = 10^(8)       # サンプル数（テスト用なので少なめ）
T = 1.0
K = 110.0
B = 100.0
mu = 0.1
r = 0.05

Random.seed!(1234)
d_normals = randn(Float64, N_STEPS * N_PATHS)

num_nodes = 100
n = num_nodes
S0_values = TCIMC.calc_cheby_abscissas(90.0, 120.0, num_nodes)
implied_vol_values = TCIMC.calc_cheby_abscissas(0.15, 0.25, num_nodes)

# インデックスを渡す設計のため、例として j = [30, 30] を利用
j = [90, 30]
h_ = 1e-1
vega_mean, vega_err = vega_asian_barrier(
    j, T, K, B,
    S0_values, implied_vol_values, mu, r,
    d_normals, N_STEPS, N_PATHS;
    h = h_
)

println("vega ≈ $vega_mean ± $vega_err")


using Statistics, Random

#num_nodes = n
Random.seed!(1234)
num_nodes_S0 = 300
S0_values_equiv = collect(range(90.0, stop=120.0, length=num_nodes_S0))
implied_vol_values_equiv = collect(range(0.15, stop=0.25, length=num_nodes))
h_delta    = S0_values_equiv[2] - S0_values_equiv[1]
Δvol   = implied_vol_values_equiv[2] - implied_vol_values_equiv[1]


# 全インデックスの組を作成して、ランダムに100点抽出
Random.seed!(1234) 
all_idx_pairs = [(i, j) for i in 1:num_nodes_S0, j in 1:num_nodes]
rand_idx_pairs = rand(vec(all_idx_pairs), 100)
# -----------------------------------
# Monte Carlo 10^8による価格評価
# -----------------------------------
vega_dict_mc_true = Dict{Tuple{Int,Int}, Float64}()
for (i_idx, j_idx) in rand_idx_pairs
    vega_mean, vega_err = vega_asian_barrier(
    [i_idx, j_idx], T, K, B,
    S0_values_equiv, implied_vol_values_equiv, mu, r,
    d_normals, N_STEPS, N_PATHS;
    h = Δvol
)
    vega_dict_mc_true[(i_idx, j_idx)] = vega_mean
end

vega_dict_mc_true

JLD2.@save "vega_dict_mc_true.jld2" vega_dict_mc_true