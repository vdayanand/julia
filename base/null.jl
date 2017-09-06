# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    Null

A type with no fields that is the type [`null`](@ref).
"""
struct Null end

"""
    null

The singleton instance of type [`Null`](@ref), used to denote a missing value.
"""
const null = Null()

"""
    Some{T}

A wrapper type used with `Union{Some{T}, Null}` to distinguish between the absence
of a value ([`null`](@ref)) and the presence of a `null` value (i.e. `Some(null)`).
It can also be used to force users of a function argument or of an object field
to explicitly handle the possibility of a field or argument being `null`,
by calling [`isnull`](@ref) and/or [`get`](@ref) before actually using the wrapped
value.
"""
struct Some{T}
    value::T
end

eltype(::Type{Some{T}}) where {T} = T

promote_rule(::Type{Some{S}}, ::Type{Some{T}}) where {S,T} = Some{promote_type(S, T)}
promote_rule(::Type{Some{T}}, ::Type{Null}) where {T} = Union{Some{T}, Null}

convert(::Type{Some{T}}, x::Some) where {T} = Some{T}(convert(T, x.value))

convert(::Type{Null}, ::Null) = null
convert(::Type{Null}, ::Any) = throw(NullException())

convert(::Type{Union{Some{T}, Null}}, x::Some) where {T} = convert(Some{T}, x)

show(io::IO, ::Null) = print(io, "null")

function show(io::IO, x::Some)
    if get(io, :compact, false)
        show(io, x.value)
    else
        print(io, "Some(")
        show(io, x.value)
        print(io, ')')
    end
end

"""
    NullException()

[`null`](@ref) was found in a context where it is not accepted.
"""
struct NullException <: Exception
end

"""
    isnull(x)

Return whether or not `x` is [`null`](@ref).
"""
isnull(x) = false
isnull(::Null) = true

"""
    get(x::Some[, y])
    get(x::Null[, y])

Attempt to access the value wrapped in `x`. Return the value if
`x` is not [`null`](@ref) (i.e. it is a [`Some`](@ref) object).
If `x` is `null`, return `y` if provided, or throw a `NullException` if not.

# Examples
```jldoctest
julia> get(Some(5))
5

julia> get(null)
ERROR: NullException()
[...]

julia> get(Some(1), 0)
1

julia> get(null, 0)
0

```
"""
function get end

get(x::Some) = x.value
get(::Null) = throw(NullException())

get(x::Some, y) = x.value
get(x::Null, y) = y
