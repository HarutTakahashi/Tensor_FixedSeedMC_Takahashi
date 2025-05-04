##########################################################
# パッケージの読み込みとエイリアスの設定
##########################################################
using Revise, LinearAlgebra, LaTeXStrings, Plots, CSV, DataFrames, JLD2, Random, Statistics, Distributions
using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
using Tensor_FixedSeedMC
import Tensor_FixedSeedMC as TCIMC

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

using Statistics, Random, Distributions

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

using Statistics

"""
    gamma_asian_barrier(
        j, T, K, B,
        S0s, sigmas, mu, r,
        d_normals, N_STEPS, N_PATHS;
        h = 1e-2
    ) -> (Float64, Float64)

アジアン・バリアオプションのGamma (∂²C/∂S0²) を中心差分近似し、
(推定値, 推定標準誤差) のタプルを返す。

- j = (row_index, col_index) のように `S0s[j[1]]` と `sigmas[j[2]]` を参照
- h: 数値微分に使う小さなステップ (S0を±hしたときの差分を用いる)

内部で `asian_barrier_payoffs` 関数を3回呼び出し、
`S0`, `S0+h`, `S0-h` の3通りの初期株価でペイオフを計算し、二次差分をとる。
"""
function gamma_asian_barrier(
    j, T, K, B,
    S0s, sigmas, mu, r,
    d_normals, N_STEPS, N_PATHS;
    h = 1e-2
) :: Tuple{Float64, Float64}

    # ① S0 を ±h ずらした初期状態を作る
    S0s_up   = copy(S0s)
    S0s_down = copy(S0s)
    # 中間(オリジナル)のS0sはそのまま
    S0s_up[j[1]]   += h
    S0s_down[j[1]] -= h

    # ② 同一の乱数列で3パターンの割引後ペイオフを計算
    pay_mid  = asian_barrier_payoffs(j, T, K, B, S0s,     sigmas, mu, r, d_normals, N_STEPS, N_PATHS)
    pay_up   = asian_barrier_payoffs(j, T, K, B, S0s_up,  sigmas, mu, r, d_normals, N_STEPS, N_PATHS)
    pay_down = asian_barrier_payoffs(j, T, K, B, S0s_down,sigmas, mu, r, d_normals, N_STEPS, N_PATHS)

    # ③ 各パスごとにガンマ (二次差分)
    #    Gamma_i = (pay_up[i] - 2 * pay_mid[i] + pay_down[i]) / (h^2)
    gammas = (pay_up .- (2 .* pay_mid) .+ pay_down) ./ (h^2)

    # ④ 平均(推定値)と標準誤差を算出
    gamma_mean = mean(gammas)
    gamma_std  = std(gammas)
    gamma_err  = gamma_std / sqrt(N_PATHS)  # 1σの標準誤差

    return gamma_mean, gamma_err
end


"""
    delta_asian_barrier(
        j, T, K, B,
        S0s, sigmas, mu, r,
        d_normals, N_STEPS, N_PATHS;
        h = 1e-2
    ) -> (Float64, Float64)

アジアン・バリアオプションのDelta (∂C/∂S0) を中心差分近似し、
(推定値, 推定標準誤差) のタプルを返す。

- j = (row_index, col_index) のように `S0s[j[1]]` と `sigmas[j[2]]` を参照
- h: 数値微分に使う小さなステップ (S0を±hしたときの差分を用いる)

内部で `asian_barrier_payoffs` 関数を2回呼び出し、
`S0+h`, `S0-h` の2通りの初期株価でペイオフを計算し、差分をとる。
"""
function delta_asian_barrier(
    j, T, K, B,
    S0s, sigmas, mu, r,
    d_normals, N_STEPS, N_PATHS;
    h = 1e-2
) :: Tuple{Float64, Float64}

    # ① S0 を ±h ずらした初期状態を作成
    S0s_up   = copy(S0s)
    S0s_down = copy(S0s)
    S0s_up[j[1]]   += h
    S0s_down[j[1]] -= h

    # ② 同一の乱数列で上方向・下方向の割引後ペイオフを計算
    pay_up   = asian_barrier_payoffs(j, T, K, B, S0s_up,   sigmas, mu, r, d_normals, N_STEPS, N_PATHS)
    pay_down = asian_barrier_payoffs(j, T, K, B, S0s_down, sigmas, mu, r, d_normals, N_STEPS, N_PATHS)

    # ③ 各パスごとにデルタ (中心差分)
    #    Delta_i = (pay_up[i] - pay_down[i]) / (2h)
    deltas = (pay_up .- pay_down) ./ (2h)

    # ④ 平均(推定値)と標準誤差を算出
    delta_mean = mean(deltas)
    delta_std  = std(deltas)
    delta_err  = delta_std / sqrt(N_PATHS)  # 1σの標準誤差

    return delta_mean, delta_err
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
delta_mean, delta_err = delta_asian_barrier(
    j, T, K, B,
    S0_values, implied_vol_values, mu, r,
    d_normals, N_STEPS, N_PATHS;
    h = h_
)

println("Delta ≈ $delta_mean ± $delta_err")


using Statistics, Random

#num_nodes = n
Random.seed!(1234)
num_nodes_S0 = 300
S0_values_equiv = collect(range(90.0, stop=120.0, length=num_nodes_S0))
implied_vol_values_equiv = collect(range(0.15, stop=0.25, length=num_nodes))
h_delta    = S0_values_equiv[2]    - S0_values_equiv[1]
Δvol   = implied_vol_values_equiv[2] - implied_vol_values_equiv[1]


# 全インデックスの組を作成して、ランダムに100点抽出
Random.seed!(1234) 
all_idx_pairs = [(i, j) for i in 1:num_nodes_S0, j in 1:num_nodes]
rand_idx_pairs = rand(vec(all_idx_pairs), 100)
# -----------------------------------
# Monte Carlo 10^8による価格評価
# -----------------------------------
delta_dict_mc_true = Dict{Tuple{Int,Int}, Float64}()
for (i_idx, j_idx) in rand_idx_pairs
    delta_mean, delta_err = delta_asian_barrier(
    [i_idx, j_idx], T, K, B,
    S0_values_equiv, implied_vol_values_equiv, mu, r,
    d_normals, N_STEPS, N_PATHS;
    h = h_delta 
)
    delta_dict_mc_true[(i_idx, j_idx)] = delta_mean
end


JLD2.@save "delta_dict_mc_true_asian_barrier.jld2" delta_dict_mc_true