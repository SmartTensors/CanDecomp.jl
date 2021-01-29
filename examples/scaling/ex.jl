using Test
using LaTeXStrings
import CanDecomp
import JLD
import PyPlot
import StaticArrays

srand(0)
A = rand(10, 3)
B = rand(5, 3)
C = rand(2, 3)

ttt = CanDecomp.totensor(A, B, C)
lrt = CanDecomp.LowRankTensor(StaticArrays.SVector(A, B, C))
tensor = CanDecomp.LowRankTensor(StaticArrays.SVector(A, B, C))
@test size(lrt) == (10, 5, 2)
@test size(permutedims(lrt, [3, 2, 1])) == (2, 5, 10)
@test permutedims(lrt, [2, 1, 3])[:, :, 2][:, :] == permutedims(ttt, [2, 1, 3])[:, :, 2]

optmethod = :nnoptim
@test CanDecomp.optim_f(C[1, :], 1, tensor[:, :, 1], StaticArrays.SVector(A, B, C), CanDecomp.tensordims(A, B, C), 0e0) ≈ 0 atol=1e-9
Cest = similar(C)
for i_3 = 1:size(C, 1)
	Cest[i_3, :] = CanDecomp.estimatecolumnoflastmatrix(i_3, tensor[:, :, i_3], StaticArrays.SVector(A, B, zeros(size(C)...)), CanDecomp.tensordims(A, B, C), Val{optmethod}; regularization=1e-3, print_level=0)
end
@show Cest, C
@test CanDecomp.full(tensor) ≈ CanDecomp.totensor(A, B, Cest) atol=1e-2
@test Cest ≈ C atol=2e-2

Cest2 = zeros(size(C)...)
matrices = StaticArrays.SVector(A, B, Cest2)
dims = CanDecomp.tensordims(A, B, C)
CanDecomp.candecompinnerloop!(matrices, tensor, dims, Val{optmethod}; regularization=1e-3, print_level=0)
@test Cest == Cest2#these should be exactly the same

noise = 0.1
Ap = A + noise * randn(size(A)...)
Bp = B + noise * randn(size(B)...)
Cp = C + noise * randn(size(C)...)
tensor_init = CanDecomp.totensor(Ap, Bp, Cp)
initerror = norm(CanDecomp.full(tensor) - tensor_init)
Ap1 = copy(Ap)
Bp1 = copy(Bp)
Cp1 = copy(Cp)
CanDecomp.candecompiteration!(StaticArrays.SVector(Ap1, Bp1, Cp1), tensor, Val{optmethod}; regularization=1e-3, print_level=0)
tensor_oneiteration = CanDecomp.totensor(Ap1, Bp1, Cp1)
oneiterationerror = norm(CanDecomp.full(tensor) - tensor_oneiteration)
@test initerror > oneiterationerror#make sure that doing an iteration actually improves things

Apf = copy(Ap)
Bpf = copy(Bp)
Cpf = copy(Cp)
print(optmethod)
@time CanDecomp.candecomp!(StaticArrays.SVector(Apf, Bpf, Cpf), tensor, Val{optmethod}; regularization=1e-3, print_level=0, max_cd_iters=25)
tensor_done = CanDecomp.totensor(Apf, Bpf, Cpf)
doneerror = norm(CanDecomp.full(tensor) - tensor_done)
@test oneiterationerror > doneerror#make sure doing more iterations improves things
@test tensor_done ≈ CanDecomp.full(tensor) rtol=2e-2

if !isfile("timings.jld")
	timingdict = Dict()
else
	timingdict = JLD.load("timings.jld", "timingdict")
end
ns = collect(5:9)
fig, ax = PyPlot.subplots()
ps = [1, 2, 4]
ts = Float64[]
for p in ps
	rmprocs(workers())
	if p > 1
		addprocs(p)
		reload("CanDecomp")
	end
	ts = Float64[]
	for n in ns
		if haskey(timingdict, (2^n, nworkers()))
			push!(ts, timingdict[(2^n, nworkers())])
		else
			A = rand(2^n, 3)
			B = rand(2^n, 3)
			C = rand(2^n, 3)
			tensor = CanDecomp.LowRankTensor(StaticArrays.SVector(A, B, C))
			Ap = (1 - noise) * A + noise * rand(size(A)...)
			Bp = (1 - noise) * B + noise * rand(size(B)...)
			Cp = (1 - noise) * C + noise * rand(size(C)...)
			print("$n: ")
			t = @elapsed @time CanDecomp.candecomp!(StaticArrays.SVector(Ap, Bp, Cp), tensor, Val{optmethod}; regularization=1e-3, print_level=0, max_cd_iters=2)
			push!(ts, t)
			timingdict[(2^n, nworkers())] = t
		end
		JLD.save("timings.jld", "timingdict", timingdict)
	end
	if nworkers() == 1
		ax[:loglog](2. ^ns, ts[end] .* (8. ^ns) ./ 8. ^ns[end], "k", basex=2, basey=2)
	end
	ax[:loglog](2. ^ns, ts, ".", basex=2, basey=2, ms=10)
end
ax[:set_ylabel]("time (s)")
ax[:set_xlabel]("N")
ax[:set_title](L"N\times N\times N"*" tensor")
ax[:legend](["perfect scaling"; map(i->i > 1 ? "$i CPUs" : "1 CPU", ps)])
display(fig)
println()
fig[:savefig]("nscaling.pdf")
PyPlot.close(fig)

fig, ax = PyPlot.subplots()
ax[:loglog](ps, ps / ps[1], "k", basex=2, basey=2)
for n in ns
	ts = Float64[]
	for p in ps
		push!(ts, timingdict[(2^n, p)])
	end
	ax[:loglog](ps, ts[1] ./ ts, ".", basex=2, basey=2, ms=10, alpha=0.5)
end
ax[:set_ylabel]("speed-up")
ax[:set_xlabel]("Number of CPUs")
ax[:set_title](L"N\times N\times N"*" tensor")
ax[:legend](["perfect scaling"; map(N->"N=$N", ns)])
display(fig)
println()
fig[:savefig]("pscaling.pdf")
PyPlot.close(fig)
