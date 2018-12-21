import CanDecomp
import Test
import Random
import StaticArrays
using LinearAlgebra

Random.seed!(0)
A = rand(10, 3)
B = rand(5, 3)
C = rand(2, 3)

T = CanDecomp.totensor(A, B, C)

@Test.testset "CanDecomp" begin

for optmethod in [:nnoptim, :nnmads, :nnjump]
	@Test.test CanDecomp.optim_f(C[1, :], 1, T[:, :, 1], StaticArrays.SVector(A, B, C), CanDecomp.tensordims(A, B, C), 0e0) ≈ 0 atol=1e-9
	Cest = similar(C)
	for i_3 = 1:size(C, 1)
		Cest[i_3, :] = CanDecomp.estimatecolumnoflastmatrix(i_3, T[:, :, i_3], StaticArrays.SVector(A, B, zeros(size(C)...)), CanDecomp.tensordims(A, B, C), Val{optmethod}; regularization=1e-3, print_level=0)
	end
	@Test.test T ≈ CanDecomp.totensor(A, B, Cest) atol=1e-2
	@Test.test Cest ≈ C atol=2e-2

	Cest2 = zeros(size(C)...)
	matrices = StaticArrays.SVector(A, B, Cest2)
	dims = CanDecomp.tensordims(A, B, C)
	CanDecomp.candecompinnerloop!(matrices, T, dims, Val{optmethod}; regularization=1e-3, print_level=0)
	@Test.test Cest == Cest2#these should be exactly the same

	noise = 0.1
	Ap = A + noise * randn(size(A)...)
	Bp = B + noise * randn(size(B)...)
	Cp = C + noise * randn(size(C)...)
	T_init = CanDecomp.totensor(Ap, Bp, Cp)
	initerror = norm(T - T_init)
	Ap1 = copy(Ap)
	Bp1 = copy(Bp)
	Cp1 = copy(Cp)
	CanDecomp.candecompiteration!(StaticArrays.SVector(Ap1, Bp1, Cp1), T, Val{optmethod}; regularization=1e-3, print_level=0)
	T_oneiteration = CanDecomp.totensor(Ap1, Bp1, Cp1)
	oneiterationerror = norm(T - T_oneiteration)
	@Test.test initerror > oneiterationerror#make sure that doing an iteration actually improves things

	Apf = copy(Ap)
	Bpf = copy(Bp)
	Cpf = copy(Cp)
	print(optmethod)
	@time CanDecomp.candecomp!(StaticArrays.SVector(Apf, Bpf, Cpf), T, Val{optmethod}; regularization=1e-3, print_level=0, max_cd_iters=25)
	T_done = CanDecomp.totensor(Apf, Bpf, Cpf)
	doneerror = norm(T - T_done)
	@Test.test oneiterationerror > doneerror#make sure doing more iterations improves things
	@Test.test T_done ≈ T rtol=2e-2
end

end
