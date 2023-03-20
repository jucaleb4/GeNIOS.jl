abstract type Solver end
abstract type ProblemData end

struct MLProblemData{T} <: ProblemData
    A
    c
    Adata::AbstractMatrix{T}
    bdata::AbstractVector{T}
    N::Int
    m::Int
    n::Int
    d2f::Function
    df::Function
    f::Function
end

struct GenericProblemData{T} <: ProblemData
    A
    c::AbstractVector{T}
    m::Int
    n::Int
    Hf::HessianOperator
    grad_f!::Function
    f::Function
    g::Function
    prox_g!::Function
end

# TODO: maybe make immutable?
mutable struct GenericSolver{T} <: Solver
    data::GenericProblemData{T} # data
    lhs_op::LinearOperator{T}   # LinerOperator for LHS of x update system
    P                           # preconditioner; TODO: combine with lhs_op?
    xk::AbstractVector{T}       # var   : primal (loss)
    Axk::AbstractVector{T}      # var   : primal 
    zk::AbstractVector{T}       # var   : primal (reg)
    zk_old::AbstractVector{T}   # var   : dual (prev step)
    uk::AbstractVector{T}       # var   : dual
    rp::AbstractVector{T}       # resid : primal
    rd::AbstractVector{T}       # resid : dual
    rp_norm::T                  # resid_norm : primal
    rd_norm::T                  # resid_norm : dual
    obj_val::T                  # log   : 0.5*||Ax - b||² + γ|x|₁
    loss::T                     # log   : 0.5*||Ax - b||² (uses zk)
    dual_gap::T                 # log   : obj_val - g(ν)
    ρ::T                        # param : ADMM penalty
    α::T                        # param : relaxation
    cache                       # cache : cache for intermediate results
end
function GenericSolver(f, grad_f!, Hf, g, prox_g!, A, c::Vector{T}; ρ=1.0, α=1.0) where {T}
    m, n = A == I ? (length(c), length(c)) : size(A)
    data = GenericProblemData(A, c, m, n, Hf, grad_f!, f, g, prox_g!)
    xk = zeros(T, n)
    Axk = zeros(T, m)
    zk = zeros(T, m)
    zk_old = zeros(T, m)
    uk = zeros(T, m)
    rp = zeros(T, m)
    rd = zeros(T, n)
    obj_val, loss, dual_gap = zero(T), zero(T), zero(T)
    rp_norm, rd_norm = zero(T), zero(T)
    cache = init_cache(data)
    
    return GenericSolver(
        data, 
        LinearOperator(A, ρ, Hf, m, n), 
        I,
        xk, Axk, zk, zk_old, uk, rp, rd, 
        rp_norm, rd_norm,
        obj_val, loss, dual_gap,
        ρ, α,
        cache)
end

function GenericSolver(data::ProblemData; ρ=1.0, α=1.0)
    return GenericSolver(
        data.f,
        data.grad_f!,
        data.Hf,
        data.g,
        data.prox_g!,
        data.A,
        data.c,
        ρ=ρ,
        α=α
    )
end

mutable struct MLSolver{T} <: Solver
    data::MLProblemData{T}      # data
    lhs_op::LinearOperator      # LinerOperator for LHS of x update system
    P                           # preconditioner; TODO: combine with lhs_op?
    xk::AbstractVector{T}       # var   : primal
    Axk::AbstractVector{T}      # var   : primal
    pred::AbstractVector{T}     # var   : primal (Adata*xk - bdata)
    zk::AbstractVector{T}       # var   : primal (reg)
    zk_old::AbstractVector{T}   # var   : dual (prev step)
    uk::AbstractVector{T}       # var   : dual
    rp::AbstractVector{T}       # resid : primal
    rd::AbstractVector{T}       # resid : dual
    rp_norm::T                  # resid_norm : primal
    rd_norm::T                  # resid_norm : dual
    obj_val::T                  # log   : 0.5*||Ax - b||² + γ|x|₁
    loss::T                     # log   : 0.5*||Ax - b||² (uses zk)
    dual_gap::T                 # log   : obj_val - g(ν)
    ρ::T                        # param : ADMM penalty
    α::T                        # param : relaxation
    λ1::T                       # param : l1 regularization
    λ2::T                       # param : l2 regularization
    cache                       # cache : cache for intermediate results
end

function MLSolver(f, df, d2f, λ1, λ2, Adata::AbstractMatrix{T}, bdata::AbstractVector{T}; ρ=1.0, α=1.0) where {T}
    N, n = size(Adata)
    #TODO: may want to add offset?
    m = n
    data = MLProblemData(-I, zeros(T, n), Adata, bdata, N, m, n, d2f, df, f)
    xk = zeros(T, n)
    Axk = zeros(T, n)
    pred = zeros(T, N)
    zk = zeros(T, m)
    zk_old = zeros(T, n)
    uk = zeros(T, n)
    rp = zeros(T, n)
    rd = zeros(T, n)
    obj_val, loss, dual_gap = zero(T), zero(T), zero(T)
    rp_norm, rd_norm = zero(T), zero(T)
    cache = init_cache(data)

    Hf = MLHessianOperator(Adata, bdata, d2f, λ2) 

    return MLSolver(
        data, 
        LinearOperator(-I, ρ, Hf, m, n),
        I,
        xk, Axk, pred, zk, zk_old, uk, rp, rd, 
        rp_norm, rd_norm,
        obj_val, loss, dual_gap,
        ρ, α, λ1, λ2,
        cache
    )
end

function init_cache(data::GenericProblemData{T}) where {T <: Real}
    m, n = data.m, data.n
    return (
        vm=zeros(T, m),
        vn=zeros(T, n),
        vn2=zeros(T, n),
        rhs=zeros(T, n),
    )
end

function init_cache(data::MLProblemData{T}) where {T <: Real}
    m, n, N = data.m, data.n, data.N
    return (
        vm=zeros(T, m),
        vn=zeros(T, n),
        vN=zeros(T, N),
        vn2=zeros(T, n),
        rhs=zeros(T, n),
    )
end


Base.@kwdef struct SolverOptions{T <: Real, S <: Real}
    relax::Bool = true
    logging::Bool = true
    indirect::Bool = false
    precondition::Bool = true
    tol::T = 1e-4
    max_iters::Int = 100
    max_time_sec::T = 1200.0
    print_iter::Int = 1
    rho_update_iter::Int = 50
    sketch_update_iter::Int = 20
    verbose::Bool = true
    multithreaded::Bool = false
    linsys_max_tol::T = 1e-1
    eps_abs::T = 1e-4
    eps_rel::T = 1e-4
    norm_type::S = 2
end


struct NysADMMLog{T <: AbstractFloat, S <: AbstractVector{T}}
    dual_gap::Union{S, Nothing}
    obj_val::Union{S, Nothing}
    iter_time::Union{S, Nothing}
    linsys_time::Union{S, Nothing}
    rp::Union{S, Nothing}
    rd::Union{S, Nothing}
    setup_time::T
    precond_time::T
    solve_time::T
end
function NysADMMLog(setup_time::T, precond_time::T, solve_time::T) where {T <: AbstractFloat}
    return NysADMMLog(
        nothing, nothing, nothing, nothing, nothing, nothing,
        setup_time, precond_time, solve_time
    )
end

function create_temp_log(::Union{GenericSolver{T}, MLSolver{T}}, max_iters::Int) where {T}
    return NysADMMLog(
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zeros(T, max_iters),
        zero(T),
        zero(T),
        zero(T)
    )
end

struct NysADMMResult{T}
    obj_val::T                 # primal objective
    loss::T                # 0.5*||Ax - b||²
    x::AbstractVector{T}       # primal soln
    z::AbstractVector{T}       # primal soln
    dual_gap::T                # duality gap
    # vT::AbstractVector{T}      # dual certificate
    log::NysADMMLog{T}
end