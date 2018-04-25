#__precompile__()

module CanDecomp
#the way we represent the canonical decomposition is like
#T[i, j, k] = sum(A[i, l] * B[j, l] * C[k, l] for l = 1:size(A, 1))
#this is based on formula 1.124 in Cichocki et al book

using Base.Cartesian
import Ipopt
import JuMP
import Mads
import Optim
import StaticArrays

macro endslice(N::Int, A, i)
	return Expr(:ref, Expr(:escape, A), [Expr(:escape, :(:)) for j=1:N - 1]..., Expr(:escape, i))
end

macro nprod(N::Int, coeffs::Expr)
	if coeffs.head != :->
		error("Second argument must be an anonymous function expression yielding a coefficient.")
	end
	cs = [Expr(:escape, Base.Cartesian.inlineanonymous(coeffs, i)) for i = 1:N]
	return Expr(:call, :*, cs...)
end

macro ngenerator(N, thing, indices)
	if indices.head != :->
		error("Third argument must be an anonymous function expression yielding a index set.")
	end
	cs = [Expr(:escape, Base.Cartesian.inlineanonymous(indices, i)) for i = 1:N]
	return Expr(:generator, Expr(:escape, thing), cs...)
end

include("lr.jl")

tensordims(matrices...) = map(x->size(x, 1), matrices)
totensor(matrices...) = totensor(StaticArrays.SVector(matrices...), tensordims(matrices...))
@generated function totensor(matrices::StaticArrays.SVector{N, T}, dims) where {N, T}
	code = quote
		facrank = size(matrices[1], 2)
		tensor = zeros(dims...)
		@nloops $N i tensor begin
			for l = 1:facrank
				(@nref $N tensor i) += @nprod $N j->matrices[j][i_j, l]
			end
		end
		return tensor
	end
	return code
end

@generated function estimatecolumnoflastmatrix(i_n, tensorslice_i_n, matrices::StaticArrays.SVector{N, T}, dims, ::Type{Val{:nnjump}}; regularization=1e0, kwargs...) where {N, T}
	q = macroexpand(:(@ngenerator $(N - 1) (((@nref $(N - 1) tensorslice_i_n i) - sum((@nprod $(N - 1) j->matrices[j][i_j, l]) * Ucol_i_n[l] for l = 1:facrank))^2) j->i_j = 1:dims[j]))
	code = quote
		m = JuMP.Model(solver=Ipopt.IpoptSolver(; kwargs...))
		facrank = size(matrices[1], 2)
		@JuMP.variable(m, Ucol_i_n[j=1:facrank], start=matrices[end][i_n, j])
		@JuMP.constraint(m, Ucol_i_n .>= 0)
		@JuMP.objective(m, Min, sum($q) + regularization * sum(Ucol_i_n[l]^2 for l=1:facrank))
		JuMP.solve(m)
		return JuMP.getvalue(Ucol_i_n)
	end
	return code
end

@generated function optim_f_lm(Ucol_i_n, i_n, tensorslice_i_n, matrices::StaticArrays.SVector{N, T}, dims, regularization) where {N, T}
	code = quote
		residuals = Array{Float64}(length(tensorslice_i_n) + length(Ucol_i_n))
		facrank = size(matrices[1], 2)
		residualindex = 1
		@nloops $(N - 1) i j->1:dims[j] begin
			thisterm = (@nref $(N - 1) tensorslice_i_n i)
			for l = 1:facrank
				thisterm -= (@nprod $(N - 1) j->matrices[j][i_j, l]) * Ucol_i_n[l]
			end
			residuals[residualindex] = thisterm
			residualindex += 1
		end
		for l = 1:facrank
			residuals[end + 1 - l] = sqrt(regularization) * Ucol_i_n[l]
		end
		return residuals
	end
	return code
end

function estimatecolumnoflastmatrix(i_n, tensorslice_i_n, matrices, dims, ::Type{Val{:nnmads}}; regularization=1e0, kwargs...)
	facrank = size(matrices[1], 2)
	f_lm = x->optim_f_lm(x, i_n, tensorslice_i_n, matrices, dims, regularization)
	f = x->optim_f(x, i_n, tensorslice_i_n, matrices, dims, regularization)
	l = 1e-15
	x0 = broadcast(max, matrices[end][i_n, :], l)
	minimizer, _ = Mads.minimize(f_lm, x0; np_lambda=1)
	negindex = minimizer .< 0
	if any(negindex)
		minimizer[negindex] = l
		minimizer, _ = Mads.minimize(f_lm, minimizer; np_lambda=1, lowerbound=l, upperbound=1e8, logtransform=true, tolX=1e-6, tolG=1e-6, tolOF=1e-32, maxEval=10000000, maxJacobians=100, sindx=0.00001)
	end
	return minimizer
end

@generated function optim_f(Ucol_i_n, i_n, tensorslice_i_n, matrices::StaticArrays.SVector{N, T}, dims, regularization) where {N, T}
	code = quote
		retval = 0.0
		facrank = size(matrices[1], 2)
		@nloops $(N - 1) i j->1:dims[j] begin
			thisterm = (@nref $(N - 1) tensorslice_i_n i)
			for l = 1:facrank
				thisterm -= (@nprod $(N - 1) j->matrices[j][i_j, l]) * Ucol_i_n[l]
			end
			retval += thisterm^2
		end
		for l = 1:facrank
			retval += regularization * Ucol_i_n[l]^2
		end
		return retval
	end
	return code
end

@generated function optim_g!(storage, Ucol_i_n, i_n, tensorslice_i_n, matrices::StaticArrays.SVector{N, T}, dims, regularization) where {N, T}
	code = quote
		retval = 0.0
		facrank = size(matrices[1], 2)
		for l = 1:facrank
			storage[l] = 2 * regularization * Ucol_i_n[l]
		end
		@nloops $(N - 1) i j->1:dims[j] begin
			thisterm = (@nref $(N - 1) tensorslice_i_n i)
			for l = 1:facrank
				thisterm -= (@nprod $(N - 1) j->matrices[j][i_j, l]) * Ucol_i_n[l]
			end
			for l = 1:facrank
				matprod = (@nprod $(N - 1) j->matrices[j][i_j, l])
				storage[l] -= 2 * thisterm * matprod
			end
		end
		return nothing
	end
	return code
end

function estimatecolumnoflastmatrix(i_n, tensorslice_i_n, matrices, dims, ::Type{Val{:nnoptim}}; regularization=1e0, kwargs...)
	facrank = size(matrices[1], 2)
	f = x->optim_f(x, i_n, tensorslice_i_n, matrices, dims, regularization)
	g! = (storage, x)->optim_g!(storage, x, i_n, tensorslice_i_n, matrices, dims, regularization)
	x0 = broadcast(max, matrices[end][i_n, :], 1e-15)
	od = Optim.OnceDifferentiable(f, g!, x0)
	lower = zeros(size(matrices[end], 2))
	upper = fill(Inf, size(matrices[end], 2))
	opt = Optim.optimize(od, x0, lower, upper, Optim.Fminbox(); show_trace=false)
	#opt = Optim.optimize(f, g!, x0, Optim.LBFGS())
	return opt.minimizer
end

function candecomp!(matrices, tensor, kind=Val{:nnjump}; done=()->false, max_cd_iters=10, kwargs...)
	i = 0
	while i < max_cd_iters && !done()
		i += 1
		candecompiteration!(matrices, tensor, kind; kwargs...)
	end
end

function candecompiteration!(matrices, tensor, kind; kwargs...)
	dims = size(tensor)
	for i = 1:length(matrices)
		perm = collect(1:length(matrices))
		perm[i] = length(matrices)#swap so that the i-th thing looks like the last thing and the last thing looks like the i-th thing
		perm[end] = i
		candecompinnerloop!(StaticArrays.SVector(matrices[perm]...), permutedims(tensor, perm), dims[perm], kind; kwargs...)
	end
end

function noclosuresallowed(chunk, matrices, dims, kind; kwargs...)
	i_n, tensorslice_i_n = chunk[1:2]
	return estimatecolumnoflastmatrix(i_n, tensorslice_i_n, matrices, dims, kind; kwargs...)
end

function candecompinnerloop!(matrices::StaticArrays.SVector, tensor, dims, kind; kwargs...)
	partialclosure = chunk->noclosuresallowed(chunk, matrices, dims, kind; kwargs...)
	candecompinnerloop!(StaticArrays.SVector(matrices...), tensor, dims, kind, partialclosure)
end

@generated function candecompinnerloop!(matrices::StaticArrays.SVector{N, T}, tensor, dims::S, kind::R, partialclosure) where {N, T, S, R}
	code = quote
		chunks = Array{Tuple{Int, Any}}(size(matrices[end], 1))
		for i = 1:size(matrices[end], 1)
			chunks[i] = (i, (@endslice $N tensor i))
		end
		Ucols = pmap(partialclosure, chunks; batch_size=ceil(Int, length(chunks) / nworkers()))
		for i = 1:size(matrices[end], 1)
			matrices[end][i, :] = Ucols[i]
		end
	end
	return code
end

const candecompdir = splitdir(splitdir(Base.source_path())[1])[1]
function test()
	include(joinpath(candecompdir, "test", "runtests.jl"))
end

end
