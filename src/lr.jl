struct LowRankTensor{N,T}
	matrices::StaticArrays.SVector{N,T}
end

mutable struct LowRankTensorEndSlice{N,T,S,M}
	matrices::StaticArrays.SVector{N,T}
	storage::Array{S,M}
	lastindex::Int
	storageisvalid::Bool
end

@generated function full(lrt::LowRankTensor{N,T}) where {N, T}
	code = quote
		tensor = zeros(eltype(T), size(lrt)...)
		facrank = size(lrt.matrices[1], 2)
		@nloops $N i tensor begin
			for l = 1:facrank
				(@nref $N tensor i) += @nprod $N j->lrt.matrices[j][i_j, l]
			end
		end
		return tensor
	end
	return code
end

function Base.getindex(lrt::LowRankTensor{N,T}, args...) where {N, T}
	if length(args) != N
		error("wrong number of indices in LowRankTensor access")
	end
	for i = 1:length(args) - 1
		if !isa(args[i], Colon)
			error("LowRankTensors only support slicing the last component")
		end
	end
	if !isa(args[end], Int)
		error("LowRankTensors only support slicing the last component")
	end
	return LowRankTensorEndSlice(lrt.matrices, Array{eltype(T)}(zeros(Int, N - 1)...), args[end], false)
end

@generated function fillinstorage!(lrt::LowRankTensorEndSlice{N,T}) where {N, T}
	code = quote
		lrt.storage = zeros(eltype(T), size(lrt)...)
		storage = lrt.storage
		facrank = size(lrt.matrices[1], 2)
		@nloops $(N - 1) i storage begin
			for l = 1:facrank
				(@nref $(N - 1) storage i) += lrt.matrices[end][lrt.lastindex, l] * (@nprod $(N - 1) j->lrt.matrices[j][i_j, l])
			end
		end
		return storage
	end
	return code
end

function Base.getindex(lrt::LowRankTensorEndSlice{N, T}, args...) where {N, T}
	if !lrt.storageisvalid
		fillinstorage!(lrt)
		lrt.storageisvalid = true
	end
	return getindex(lrt.storage, args...)
end

@generated function Base.size(lrt::LowRankTensor{N,T}) where {N, T}
	code = quote
		return @ntuple $N i->size(lrt.matrices[i], 1)
	end
	return code
end

@generated function Base.size(lrt::LowRankTensorEndSlice{N,T}) where {N, T}
	code = quote
		return @ntuple $(N - 1) i->size(lrt.matrices[i], 1)
	end
	return code
end

function Base.permutedims(tensor::LowRankTensor, perm)
	return LowRankTensor(StaticArrays.SVector(tensor.matrices[perm]...))
end
