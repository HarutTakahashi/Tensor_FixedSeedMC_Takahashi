using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
# =============================================================================
# 1. Helper Functions: Chebyshev Abscissas, Basis, and Its Derivative
# =============================================================================

function calc_cheby_abscissas(a::Float64, b::Float64, n::Int)
    @assert b > a "calc_cheby_abscissas: need a < b"
    @assert n >= 1 "calc_cheby_abscissas: n < 1"
    x = Vector{Float64}(undef, n)
    bma = 0.5*(b - a)
    bpa = 0.5*(b + a)
    for k in 0:(n-1)
        y = cos(pi*(k + 0.5)/n)
        x[k+1] = y*bma + bpa
    end
    return reverse(x)
end

function chebyshev_basis(a::Float64, b::Float64, x::Float64, N::Int)
    # (必要に応じて境界チェックする場合は残す or 削除)
    # if x < a || x > b
    #     error("x is out of bounds")
    # end

    # -1 <= y <= 1 に収める
    y_raw = (2*x - (a + b)) / (b - a)
    y = clamp(y_raw, -1.0, 1.0)

    theta = acos(y)
    k = 0:(N-1)
    T = cos.(theta .* k)

    # Chebyshev 係数 (通常 T_0 は 1 と定義されるが、
    # 「基底ベクトルの重み付け」として 1/2 を使う定義もある)
    scale = ones(Float64, N)
    scale[1] = 0.5
    return scale .* T
end


function chebyshev_basis_derivative(a::Float64, b::Float64, x::Float64, N::Int)
    # x が [a,b] の外ならエラー
    if x < a || x > b
        error("x is out of bounds")
    end

    # -1 ≤ y ≤ 1 のはずが、浮動小数点誤差でわずかに範囲外になることがある
    y = (2*x - (a+b)) / (b-a)
    # クランプしてから acos をとる
    y_clamped = clamp(y, -1.0, 1.0)
    theta = acos(y_clamped)

    c = 2/(b-a)
    # k=0の場合は微分結果は0, k>0の場合は k*c/sqrt(1-y^2)*sin(kθ)
    return [
        (k == 0 ? 0.0 : k * c/sqrt(1 - y_clamped^2) * sin(k*theta))
        for k in 0:(N-1)
    ]
end

# =============================================================================
# 2. Tensor Train Evaluation Functions (for TCI method)
# =============================================================================



function compute_left_cache(tt::TCI.TensorTrain{T}, weights::Vector{Vector{Float64}}, c::Int) where T
    L = ones(T, 1, 1)
    for s in 1:(c-1)
        r, n, r_next = size(tt.sitetensors[s])
        LMv = L * reshape(tt.sitetensors[s], r, n*r_next)
        LM = reshape(LMv, n, r_next)
        ws = reshape(weights[s], 1, n)
        L = ws * LM
    end
    return L
end

function compute_right_cache(tt::TCI.TensorTrain{T}, weights::Vector{Vector{Float64}}, c::Int) where T
    N = length(tt.sitetensors)
    R = ones(T, 1, 1)
    for s in N:-1:(c+1)
        r, n, r_next = size(tt.sitetensors[s])
        MRv = reshape(tt.sitetensors[s], r*n, r_next) * R
        MR = reshape(MRv, r, n)
        ws = reshape(weights[s], n, 1)
        R = MR * ws
    end
    return R
end

function update_center(tt::TCI.TensorTrain{T}, left_cache::Array{T,2}, right_cache::Array{T,2},
                       new_weight::Vector{Float64}, c::Int) where T
    M_c = tt.sitetensors[c]
    r_prev, n, r_next = size(M_c)
    LMv = left_cache * reshape(M_c, r_prev, n*r_next)
    LM = reshape(LMv, n, r_next)
    ws_new = reshape(new_weight, 1, n)
    center = ws_new * LM
    final = center * right_cache
    return final[1,1]
end


# Optionally, compute Chebyshev coefficients via TCI evaluation
function compute_chebyshev_coeff(tt::TCI.TensorTrain{T}, A::Vector) where T
    new_cores = Vector{Array{T,3}}(undef, length(tt.sitetensors))
    for i in 1:length(tt.sitetensors)
        core = tt.sitetensors[i]
        r1, n, r2 = size(core)
        p, nA = size(A[i])
        @assert n == nA "Matrix A の列数とコアのサイズ不一致"
        new_core = zeros(T, r1, p, r2)
        for i1 in 1:r1, i3 in 1:r2
            new_core[i1, :, i3] = A[i] * core[i1, :, i3]
        end
        new_cores[i] = new_core .* (2.0/n)
    end
    return TCI.TensorTrain{T}(new_cores)
end


function compute_chebyshev_coeff(tt::TCI.TensorTrain{T}, A::Vector{<:AbstractMatrix{T}}) where T
    # Here, specify that each element of new_cores is Array{Float64,3}:
    new_cores = Vector{Array{T,3}}(undef, length(tt.sitetensors))
    
    for i in 1:length(tt.sitetensors)
        core = tt.sitetensors[i]
        r1, n, r2 = size(core)
        p, nA    = size(A[i])
        @assert n == nA "Matrix A の列数とコアのサイズ不一致"
        
        new_core = zeros(T, r1, p, r2)
        for i1 in 1:r1, i3 in 1:r2
            new_core[i1, :, i3] = A[i] * core[i1, :, i3]
        end
        
        # Scale by 2.0/n
        new_cores[i] = new_core .* (2.0/n)
    end
    
    return TCI.TensorTrain(new_cores)
end

# 2D DCT to obtain Chebyshev coefficients
function dct_matrix(N::Int)
    @assert N >= 1 "dct_matrix: N must be >= 1"
    M = zeros(Float64, N, N)
    for k in 0:(N-1), j in 0:(N-1)
        M[k+1, j+1] = cos(pi * k * (j + 0.5) / N)
    end
    return reverse(M, dims=2)
end


# (2) チェビシェフスペクトル微分法
function cheb_nodes_and_diffmat(N)
    j = 0:N
    x = cos.(pi .* j ./ N)
    c = [2; ones(N-1); 2] .* (-1).^j
    X = repeat(x, 1, N+1)
    dX = X .- X'
    # I を明示的な行列に変換して加算
    D = (c * (1 ./ c)') ./ (dX .+ Matrix{Float64}(I, size(dX)...))
    D = D .- diagm(0 => sum(D, dims=2)[:])
    return x, D
end


# 2階微分: d²/dx² cos(kθ)
function chebyshev_basis_second_derivative(a::Float64, b::Float64, x::Float64, N::Int)
    if x < a || x > b
        error("x is out of bounds")
    end
    y = (2*x - (a+b))/(b-a)
    theta = acos(y)
    c = 2/(b-a)
    A = c/sqrt(1 - y^2)                   # A(x) = c/√(1-y²)
    Aprime = c^2 * y/( (1-y^2)^(3/2) )      # A'(x) = c²*y/(1-y²)^(3/2)
    return [ (k == 0 ? 0.0 : k * (Aprime * sin(k*theta) - k * A^2 * cos(k*theta)))
             for k in 0:(N-1) ]
end