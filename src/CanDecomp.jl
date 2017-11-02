module CanDecomp

#the way we represent the canonical decomposition is like
#T[i, j, k] = sum(A[l, i] * B[l, j] * C[l, k] for l = 1:size(A, 1))
#this is based on formula 1.124 in Cichocki et al book

using Base.Cartesian
import Ipopt
import JuMP
import StaticArrays

macro nprod(N::Int, coeffs::Expr)
	if coeffs.head != :->
		error("Second argument must be an anonymous functino expression yielding a coefficient.")
	end
	cs = [Expr(:escape, Base.Cartesian.inlineanonymous(coeffs, i)) for i = 1:N]
	return Expr(:call, :*, cs...)
end

macro ngenerator(N, thing, indices)
	if indices.head != :->
		error("Third argument must be an anonymous functino expression yielding a index set.")
	end
	cs = [Expr(:escape, Base.Cartesian.inlineanonymous(indices, i)) for i = 1:N]
	return Expr(:generator, Expr(:escape, thing), cs...)
end

tensordims(matrices...) = map(x->size(x, 2), matrices)
totensor(matrices...) = totensor(StaticArrays.SVector(matrices...), tensordims(matrices...))
@generated function totensor(matrices::StaticArrays.SVector{N, T}, dims) where {N, T}
	code = quote
		facrank = size(matrices[1], 1)
		tensor = zeros(dims...)
		@nloops $N i tensor begin
			for l = 1:facrank
				(@nref $N tensor i) += @nprod $N j->matrices[j][l, i_j]
			end
		end
		return tensor
	end
	return code
end

function estimatethirdmatrixcomponent(i_3, tensorslice_i_3, matrices::StaticArrays.SVector, dims; regularization=1e0, kwargs...)
	m = JuMP.Model(solver=Ipopt.IpoptSolver(; kwargs...))
	facrank = size(matrices[1], 1)
	@JuMP.variable(m, Ucol_i_3[j=1:facrank], start=matrices[3][j])
	@JuMP.constraint(m, Ucol_i_3 .>= 0)
	@JuMP.objective(m, Min, sum((tensorslice_i_3[i_1, i_2] - sum(matrices[1][l, i_1] * matrices[2][l, i_2] * Ucol_i_3[l] for l=1:facrank))^2 for i_1 = 1:dims[1] for i_2 = 1:dims[2]) + regularization * sum(Ucol_i_3[l]^2 for l=1:facrank))
	JuMP.solve(m)
	return JuMP.getvalue(Ucol_i_3)
end

@generated function estimatecomponentoflastmatrix(i_n, tensorslice_i_n, matrices::StaticArrays.SVector{N, T}, dims; regularization=1e0, kwargs...) where {N, T}
	q = macroexpand(:(@ngenerator $(N - 1) (((@nref $(N - 1) tensorslice_i_n i) - sum((@nprod $(N - 1) j->matrices[j][l, i_j]) * Ucol_i_n[l] for l = 1:facrank))^2) j->i_j = 1:dims[j]))
	code = quote
		m = JuMP.Model(solver=Ipopt.IpoptSolver(; kwargs...))
		facrank = size(matrices[1], 1)
		@JuMP.variable(m, Ucol_i_n[j=1:facrank], start=matrices[end][j])
		@JuMP.constraint(m, Ucol_i_n .>= 0)
		@JuMP.objective(m, Min, sum($q) + regularization * sum(Ucol_i_n[l]^2 for l=1:facrank))
		JuMP.solve(m)
		return JuMP.getvalue(Ucol_i_n)
	end
	return code
end

end
