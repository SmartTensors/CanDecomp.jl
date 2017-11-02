import CanDecomp

A = rand(3, 10)
B = rand(3, 5)
C = rand(3, 2)

tensor = CanDecomp.totensor(A, B, C)

Cest = similar(C)
Cest2 = similar(C)
for i_3 = 1:size(C, 2)
	Cest[:, i_3] = CanDecomp.estimatethirdmatrixcomponent(i_3, tensor[:, :, i_3], StaticArrays.SVector(A, B, zeros(size(C)...)), CanDecomp.tensordims(A, B, C); regularization=1e-3, print_level=0)
	Cest2[:, i_3] = CanDecomp.estimatecomponentoflastmatrix(i_3, tensor[:, :, i_3], StaticArrays.SVector(A, B, zeros(size(C)...)), CanDecomp.tensordims(A, B, C); regularization=1e-3, print_level=0)
end
@show vecnorm(tensor - CanDecomp.totensor(A, B, Cest)) / vecnorm(tensor)
@show vecnorm(tensor - CanDecomp.totensor(A, B, Cest2)) / vecnorm(tensor)
