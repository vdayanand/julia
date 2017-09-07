# This file is a part of Julia. License is MIT: https://julialang.org/license

adjoint(a::AbstractArray) = error("adjoint not defined for $(typeof(a)). Consider using `permutedims` for higher-dimensional arrays.")

## Matrix transposition ##

"""
    adjoint!(dest,src)

Conjugate transpose array `src` and store the result in the preallocated array `dest`, which
should have a size corresponding to `(size(src,2),size(src,1))`. No in-place transposition
is supported and unexpected results will happen if `src` and `dest` have overlapping memory
regions.
"""
adjoint!(B::AbstractMatrix, A::AbstractMatrix) = transpose_f!(adjoint, B, A)
function adjoint!(B::AbstractVector, A::AbstractMatrix)
    indices(B,1) == indices(A,2) && indices(A,1) == 1:1 || throw(DimensionMismatch("transpose"))
    adjointcopy!(B, A)
end
function adjoint!(B::AbstractMatrix, A::AbstractVector)
    indices(B,2) == indices(A,1) && indices(B,1) == 1:1 || throw(DimensionMismatch("transpose"))
    adjointcopy!(B, A)
end

function adjointcopy!(B, A)
    RB, RA = eachindex(B), eachindex(A)
    if RB == RA
        for i = RB
            B[i] = adjoint(A[i])
        end
    else
        for (i,j) = zip(RB, RA)
            B[i] = adjoint(A[j])
        end
    end
end

function adjoint(A::AbstractMatrix)
    ind1, ind2 = indices(A)
    B = similar(A, adjoint_type(eltype(A)), (ind2, ind1))
    adjoint!(B, A)
end
