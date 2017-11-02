using Base.Test
import CanDecomp

srand(0)
A = rand(3, 10)
B = rand(3, 5)
C = rand(3, 2)

tensor = CanDecomp.totensor(A, B, C)

Cest = similar(C)
for i_3 = 1:size(C, 2)
	Cest[:, i_3] = CanDecomp.estimatecolumnoflastmatrix(i_3, tensor[:, :, i_3], StaticArrays.SVector(A, B, zeros(size(C)...)), CanDecomp.tensordims(A, B, C); regularization=1e-3, print_level=0)
end
@test tensor ≈ CanDecomp.totensor(A, B, Cest) atol=1e-3
@test Cest ≈ C atol=1e-3

Cest2 = zeros(size(C)...)
matrices = StaticArrays.SVector(A, B, Cest2)
dims = CanDecomp.tensordims(A, B, C)
CanDecomp.candecompinnerloop!(matrices, tensor, dims; regularization=1e-3, print_level=0)
@test Cest == Cest2#these should be exactly the same

noise = 0.1
Ap = A + noise * randn(size(A)...)
Bp = B + noise * randn(size(B)...)
Cp = C + noise * randn(size(C)...)
tensor_init = CanDecomp.totensor(Ap, Bp, Cp)
initerror = vecnorm(tensor - tensor_init)
Ap1 = copy(Ap)
Bp1 = copy(Bp)
Cp1 = copy(Cp)
CanDecomp.candecompiteration!(StaticArrays.SVector(Ap1, Bp1, Cp1), tensor; regularization=1e-3, print_level=0)
tensor_oneiteration = CanDecomp.totensor(Ap1, Bp1, Cp1)
oneiterationerror = vecnorm(tensor - tensor_oneiteration)
@test initerror > oneiterationerror#make sure that doing an iteration actually improves things

Apf = copy(Ap)
Bpf = copy(Bp)
Cpf = copy(Cp)
@time CanDecomp.candecomp!(StaticArrays.SVector(Apf, Bpf, Cpf), tensor; regularization=1e-3, print_level=0, max_cd_iters=25)
tensor_done = CanDecomp.totensor(Apf, Bpf, Cpf)
doneerror = vecnorm(tensor - tensor_done)
@test oneiterationerror > doneerror#make sure doing more iterations improves things
@test tensor_done ≈ tensor rtol=1e-2