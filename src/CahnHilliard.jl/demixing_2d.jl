import DelimitedFiles: writedlm, readdlm
using Printf
using LinearAlgebra
using FFTW
using Random
import ForwardDiff: hessian
using Distributions
## Plotting tools
using StatsPlots
using Plots
pyplot()
theme(:ggplot2)
## Integrated python
using PyCall
global np = pyimport("numpy")

## Constants
const LENGTH = 256
const SEED   = false
const SAVE   = false

const max_count = 10000#25000
const save_interval = 50

global FLAG_newsave = false

## Variables type
struct Variables
    δ::Dict
    p̄::AbstractFloat
    θ::Dict
    M::AbstractFloat # for now
end
############################################
""" Calculate centered finite-difference approximation of Laplacian ∇² """
function fdm_laplacian(x, δx)
    offset_i₊, offset_i₋ = [x[2:end,:];x[1:1,:]], [x[end:end,:];x[1:end-1,:]]
    offset_j₊, offset_j₋ = [x[:,2:end] x[:,1:1]], [x[:,end:end] x[:,1:end-1]]
    return (-4.0*x .+ offset_i₊ .+ offset_i₋ .+ offset_j₋ .+ offset_j₊)/(δx^2)
end
############################################
""" Randomly initialise ϕ(x) ~ U[p̄ - δᵩ/2,p̄ + δᵩ/2] and μ(x) = 0 """
function initialise(params; seed::Union{Missing,Integer}=missing)
    if ismissing(seed)
        rng = Random.MersenneTwister()
    else
        rng = Random.MersenneTwister(seed)# fix random seed
    end
    if SAVE FLAG_newsave = true end

    ϕ = params.p̄ .+ (rand(rng,Float64,LENGTH,LENGTH) .- 0.5).*params.δ[:ϕ]
    μ = zeros(Float64, LENGTH, LENGTH)

    return ϕ, μ
end
###########################################
""" Finite difference interation of dϕ/dt
        ϕₜ₊₁ = ϕₜ + Mδₜ∇²μ
    Updates in-place
"""
function diffuse_ϕ!(ϕ, μ, params)
    M, δ = params.M, params.δ

    ∇²μ = fdm_laplacian(μ, δ[:x])
    ϕ .+= M.*δ[:t].*∇²μ
end
############################################
""" Calculate μ = dF/dϕ
        μ = -αϕ + βϕ³ - κ∇²ϕ
"""
function calculate_μ!(μ, ϕ, params)
    θ, δ = params.θ, params.δ

    ∇²ϕ = fdm_laplacian(ϕ, δ[:x])
    μ .= -θ[:α].*ϕ .+ θ[:β].*ϕ.^3 .- θ[:κ].*∇²ϕ
end
############################################
""" Iterate calculation of ϕ(T) in place """
function calculate_ϕ!(ϕ, μ, params; num_iterations::Integer = 1e4)
    for k ∈ 1:max_count
        calculate_μ!(μ, ϕ, params)
        diffuse_ϕ!(ϕ, μ, params)
        if any(isnan.(ϕ))
            error(@sprintf("found nan at k=%d",k))
        end
        if SAVE && ((k-1) % save_interval == 0)
            write_to_file(ϕ, μ, @sprintf("_%d", k))
        end
    end
end
#############################################
""" Write ϕ, μ to file for later plotting """
function write_to_file(ϕ, μ, postfix::String = "")
    @printf("writing to file %s... ", postfix)

    filename_μ = @sprintf("phasesep2d\\mu%s.dat", postfix)
    open(filename_μ, "w") do fid
        writedlm(fid, μ, ' ')
    end

    filename_ϕ = @sprintf("phasesep2d\\phi%s.dat", postfix)
    open(filename_ϕ, "w") do fid
        writedlm(fid, ϕ, ' ')
    end
end
##############################################
""" Create gif of simulation from save files """
function animate(filename="2d_anim.gif"; fps::Integer = 15)
    print("creating animation")
    anim = @animate for i ∈ 1:save_interval:max_count
        ϕk = readdlm(@sprintf("phasesep2d\\phi_%s.dat", i))
        contourf(ϕk)
    end
    gif(anim, filename, fps=fps)
end
##############################################
""" Generate Variables construct containing parameters
    Takes in dictionary of priors or values for α, β, κ
"""
function generate_params(priors)
    # Discrete meshes for space, time and ϕ
    δ = default_params.δ
    # Mean mixing p̄
    p̄ = default_params.p̄
    # Sample θ from priors
    θ = sample_free_energy_parameters(priors)
    # Diffusion field / constant
    M = default_params.M
    return Variables(δ, p̄, θ, M)
end
##############################################
""" Sample free energy parameters from dictionary of priors """
function sample_free_energy_parameters(priors::Dict{Symbol, <:Sampleable})
    # dF(x)/dφ = -αφ + βφ³ - κ∇²φ
    return Dict([first(kv) => rand(last(kv)) for kv ∈ priors])
end
""" Return free energy parameters from dictionary of values """
function sample_free_energy_parameters(priors::Dict{Symbol, <:AbstractFloat})
    return Dict([first(kv) => last(kv) for kv ∈ priors])
end
##############################################
##############################################
const default_params = Variables(
    Dict(:x => 5e2, :t => 1e8, :ϕ => 2e-1),
    0.,
    Dict(:α => 7.07e-6, :β => 1.2e-4, :κ => 0.71),
    1.
)
##############################################
""" calculate S(q,t) given at nt uniformly spaced discrete times in num_iterations """
function calculate_S(ϕ, μ, params; nt=9, num_iterations=1e4)

    # Discrete times
    ts = round.(Int, range(0, num_iterations, length=nt))

    n = 32 # what's this ?

    # Get magnitude of frequency [should this be |F|² or |F| ?]
    F(φ) = fft(φ .- params.p̄) |> fftshift .|> abs

    # Calculate radial average of spectrum at time 0
    q, S₀ = radial_avg(F(ϕ), params)

    # Preallocate data struct
    S = zeros(length(S₀), nt)
    S[:,1] = S₀

    t = 2 # Start from index 2 (where ts[1] = 0)

    # Iterate calculation of φ
    for k ∈ 1:num_iterations
        # Update φ
        calculate_μ!(μ, ϕ, params)
        diffuse_ϕ!(ϕ, μ, params)
        # Error checking !
        if any(isnan.(ϕ))
            error(@sprintf("found nan at k=%d",k))
        end
        # If file saving (for animation)
        if SAVE && ((k-1) % save_interval == 0)
            write_to_file(ϕ, μ, @sprintf("_%d", k))
        end

        if k ∈ ts # calculate S(q,t)
            Fᵩ = F(ϕ) # absolute frequency
            _, Sₜ  = radial_avg(Fᵩ, params)
            S[:,t] = Sₜ
            t+=1 # increment time index
        end
    end
    return ts, q, S
end
##############################################
""" Radial average of 0-centered frequency
    > Uses python : np.bincount to calculate average radial frequency
"""
function radial_avg(Fᵩ, params)
    n = size(Fᵩ,1)
    # Create index map
    indices = n-> map(i -> getindex.(CartesianIndices((n,n)), i), [1,2])
    xi, yi = indices(n)
    # Calculate integer distance matrix
    r = sqrt.((xi .- .5n).^2 + (yi .- .5n).^2) .|> round .|> Int
    # Calculate frequency map Fᵩ = F[φ](||ω||)
    qs = range(-1. / 2params.δ[:x], 1. / 2params.δ[:x], length=n)
    qr = [sqrt.((qx^2 + qy^2)) for qx ∈ qs, qy ∈ qs]
    # Bin frequencies
    q = np.bincount(r[:], qr[:]) ./ np.bincount(r[:])
    S = np.bincount(r[:], Fᵩ[:]) ./ np.bincount(r[:])
    return q, S
end
##############################################
""" [DEPRECATED] Discrete Fourier transform F[ϕ](qx,qy)"""
function H(φ, qx, qy)
    m = size(φ,1)
    δx = params.δ[:x]
    h = 0
    for j ∈ 0:m-1
        for k ∈ 0:m-1
            ω = -2π*(j*qx+k*qy)/m
            h += φ[j+1,k+1] .* complex(cos(ω),sin(ω))
        end
    end
    return h/m
end
#############################################
# Export default parameter dictionary
params = Dict(
    :α => LogNormal(-11.86, 0.1),
    :β => Normal(1.2e-4, 1e-6),
    :κ => LogNormal(-0.35, 0.01)
)
#############################################
""" Generate n samples given fixed parameter values or priors for α,β,κ """
function generate_samples(n=10; params::Union{Missing,Dict} = missing, seed::Union{Missing, Integer}=missing)
    Ss = []
    local q
    local ts
    for i ∈ 1:n
        ts, q, s = main(toplot=false, params=params, seed=seed)
        push!(Ss, s)
    end
    return ts, q, Ss
end
##############################################
const main(args...; kwargs...) = run_simulation(args...; kwargs...)
""" Initalise and run a simulation given an optional parameter dictionary """
function run_simulation(;toplot=false, params::Union{Missing,Dict} = missing, seed::Union{Missing, Integer}=missing)
    if ismissing(params) # Defer to default values
        params = Dict(
            :α => 7.07e-6,
            :β => 1.2e-4,
            :κ => 0.71
        ) |> generate_params
    else
        params = params |> generate_params
    end

    # Initialise
    ϕ, μ = initialise(params; seed=seed)
    # Calculate S(q,t;θ)
    ts, q, S = calculate_S(ϕ,μ,params)

    if toplot # plot S(q,t;θ)
        p = plot(yscale=:log)
        for (i,t) ∈ enumerate(ts)
            plot!(p,
                q, S[:,i],
                linealpha=0., markershape=:circle,
                label=t/params.δ[:t]
            )
        end
        display(p)
    end

    return ts, q, S
end

####################################################
# Plotting utilities
#################################
""" Plot confidence intervals for samples of S(q,t) """
function plot_err(ts, q, Ss; cname=:blues, yscale=:log, kwargs...)
    # Get mean S(q,t) over all simulations
    x̄ = [mean([s[i,t] for s ∈ Ss]) for t ∈ 1:9, i ∈ 1:size(Ss[1],1)]
    # Calculate 95% confidence intervals
    lci = [quantile([s[i,t] for s ∈ Ss],0.025) for t ∈ 1:9, i ∈ 1:size(Ss[1],1)]
    uci = [quantile([s[i,t] for s ∈ Ss],0.975) for t ∈ 1:9, i ∈ 1:size(Ss[1],1)]

    # Generate colour map
    cmap = cgrad(cname) |> (g-> RGB[g[z] for z in range(0,1,length=length(ts))])

    p = plot(;yscale=yscale, kwargs...) #
    for i ∈ 1:length(ts)
        plot!(p, q, x̄[i,:], linealpha=0.5, linewidth=1., linecolor=cmap[i], label=ts[i]) # Plot mean
        plot!(p, q, uci[i,:], fill=(lci[i,:],0.2, cmap[i]), linealpha=0., label="") # Shade confidence interval
    end
    return p # <: Plots.Plot
end
#########
""" Plot individual samples of S(q,t) """
function plot_samples(ts, q, Ss; cname=:blues, yscale=:log, kwargs...)
    # Generate colour map
    cmap = cgrad(cname) |> (g-> RGB[g[z] for z in range(0,1,length=length(ts))])

    p = plot(; yscale=yscale, kwargs...) #
    for (k,sample) ∈ enumerate(Ss) # for each sample
        for (i,t) ∈ enumerate(ts)     # plot S(q,t) for t = 0...T
            plot!(
                p, q, sample[:,i],
                line=(0.1, 1, cmap[i]),
                label= k > 1 ? "" : string(t)
            )
        end
    end
    return p # <: Plots.Plot
end
###################################
""" Show boxplot or violin plot of errors between true and false samples and y """
function compare_loss(y, true_samples, false_samples; metric=rmse, type=:violin, kwargs...)
    ℓ_true  = calculate_loss(y, true_samples, metric=metric)
    ℓ_false = calculate_loss(y, false_samples, metric=metric)

    plot_func! = type == :violin ? violin! : boxplot!

    p = plot(xticks=([0, 1], ["correct", "incorrect"]))
    plot_func!(p, [0], ℓ_true, fillcolor=:blue, fillalpha=0.5, label="")
    plot_func!(p, [1], ℓ_false, fillcolor=:red, fillalpha=0.5, label="")
    return p
end
####################################
### Score metrics
############
function square_error(y, ỹ)
    """ (y - ỹ)² """
    sqdiff = (y .- ỹ).^2
    return sqdiff[:]
end

mse(y, ỹ) = square_error(y,ỹ) |> mean
rmse(y, ỹ) = mse(y,ỹ) |> sqrt
sse(y, ỹ) = square_error(y,ỹ) |> sum

#################################
""" For array of samples, calculate loss compared to y """
function calculate_loss(y, samples; metric=mse)
    ℓ = []
    for sample ∈ samples
        push!(ℓ, metric(y, sample))
    end
    return ℓ
end
""" For array of samples, calculate loss compared to y
    > utility to take data directly from generate_samples()
"""
function calculate_loss(y::Tuple, samples::Tuple; kwargs...)
    calculate_loss(y[3][1], samples[3]; kwargs...)
end
#################################
""" Verification code """
function verify_metric(θ₁::Dict, θ₂::Dict; metric=rmse, N=50)
    y1 = generate_samples(1, params=θ₁)
    y2 = generate_samples(1, params=θ₂)

    samples_1 = generate_samples(N, params=θ₁)
    samples_2 = generate_samples(N, params=θ₂)

    p1 = plot_samples(samples_1..., cname=:viridis)
    p2 = plot_samples(samples_2..., cname=:viridis)

    plot(p1, p2, sharey=true, layout=(1,2)) |> display

    p1 = compare_loss(y1, samples_1, samples_2, metric=metric)
    p2 = compare_loss(y2, samples_1, samples_2, metric=metric)

    plot(p1, p2, sharey=true, layout=(1,2)) |> display
    return (y1, samples_1), (y2, samples_2)
end

""" This is an example """
function example_comparison()
    θ₁ = Dict(
        :α => LogNormal(-11.86, 0.1),
        :β => Normal(1.2e-4, 1e-6),
        :κ => LogNormal(-0.35, 0.01)
    )
    θ₂ = Dict(
        :α => LogNormal(-11.0, 0.1),
        :β => Normal(1.2e-4, 1e-6),
        :κ => LogNormal(-0.35, 0.01)
    )
    return verify_metric(θ₁, θ₂, metric=rmse)
end

####################################
### Free energy utilities (not used by main code)
""" Free energy F(ϕ;θ) """
function free_energy(ϕ, α, β)
    y = -.5α*ϕ.^2 + .25β*ϕ.^4
    return y
end

""" Gradient of free energy dF/dϕ """
function free_energy_grad(ϕ, α, β)
    y = -α*ϕ + β*ϕ.^3
    return y
end

""" Plot F(ϕ) or dF/dϕ """
function plot_free_energy(params; gradient::Bool = false)
    n₁,n₂ = 100, 100
    ϕ = range(-0.4, 0.4, length=n₁)

    func = gradient ? free_energy_grad : free_energy

    samples = [func(ϕ, rand(params[:α]), rand(params[:β])) for _ ∈ 1:n₂]

    p1 = plot()
    [plot!(p1, ϕ, fᵢ, alpha=0.1, linecolor=:black, label="") for fᵢ ∈ samples]

    summ(func::Function) = [func([samples[i][k] for i ∈ 1:n₂]) for k ∈ 1:n₁]

    μ, lci, uci = summ(mean), summ(x->quantile(x,0.025)), summ(x->quantile(x,0.975))
    p2 = plot(ϕ, μ, linewidth=1., linecolor=:blue)
    plot!(p2, ϕ, uci, fill=(lci, 0.25, :blue), linealpha=0.)
    return plot(p1,p2,layout=(2,1))
end
plot_free_energy_grad(params) = plot_free_energy(params, gradient=true)

############
# Some verification code for numerics of Laplacian approximation
########
function verify_numerics(; f = missing, options = (:show_stats,))
    δx = 1/32

    xs     = range(-2., 2., step=δx)
    test_f = ismissing(f) ? x -> exp.(-x'*x/0.5)/(π/2.) : f
    true_∇²f = ismissing(f) ? x -> (x'*x-0.5)*exp(-x'*x/0.5)/(2π*0.015625) : x-tr(hessian(test_f, x))

    numeric_laplacian(f) = fdm_laplacian([f([x,y]) for x ∈ xs, y ∈ xs], δx)
    auto_laplacian(f)    = [tr(hessian(f, [x,y])) for x ∈ xs, y ∈ xs]

    ∇²true = [true_∇²f([x,y]) for x ∈ xs, y ∈ xs]
    ∇²numr = numeric_laplacian(test_f)
    ∇²auto = auto_laplacian(test_f)

    maxs    = (maximum(∇²numr), maximum(∇²auto), maximum(∇²true))
    mins    = (minimum(∇²numr), minimum(∇²auto), minimum(∇²true))
    diff    = ∇²numr .- ∇²true
    rel_err = (∇²numr ./ ∇²true) .- 1.

    if :show_stats ∈ options
        @printf("        |    numeric  |  auto      |   true\n")
        @printf("max val |   %7.4f   |  %7.4f   |  %7.4f\n", maxs...)
        @printf("min val |   %7.4f   |  %7.4f   |  %7.4f\n", mins...)
        @printf("--------|    min      |  median    | max\n")
        @printf("abs dif |  %7.4f    |  %7.4f   | %7.4f\n",
                minimum(abs.(diff)), median(abs.(diff)), maximum(abs.(diff))
        )
        @printf("sqr dif |  %7.4f    |  %7.4f   | %7.4f\n",
                minimum(diff.^2), median(diff.^2), maximum(diff.^2)
        )
        @printf("rel err |  %7.4f    |  %7.4f   | %7.4f\n",
                minimum(rel_err), median(rel_err), maximum(rel_err)
        )
    end
    if :show_plots ∈ options
        #
    end
end
