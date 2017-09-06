# This file is a part of Julia. License is MIT: https://julialang.org/license

## eltype()

@test eltype(Some(1)) === Int

## promote()

@test promote_type(Some{Int}, Some{Float64}) === Some{Float64}

## convert()

# These conversions must fail to prevent ambiguities
# when a value to wrap is already a Some or a Null
@test_throws MethodError convert(Some, 1)
@test_throws MethodError convert(Union{Some, Null}, 1)
@test_throws MethodError convert(Some{Int}, 1)
@test_throws MethodError convert(Union{Some{Int}, Null}, 1)

@test convert(Some, Some(1)) === convert(Union{Some, Null}, Some(1)) === Some(1)
@test convert(Some{Int}, Some(1)) === convert(Union{Some{Int}, Null}, Some(1)) === Some(1)
@test convert(Some{Int}, Some(1.0)) === convert(Union{Some{Int}, Null}, Some(1.0)) === Some(1)

@test_throws MethodError convert(Some, null)
@test_throws MethodError convert(Some{Int}, null)

@test_throws NullException convert(Null, 1)
@test_throws NullException convert(Null, Some(1))

@test convert(Some, Some(null)) === Some(null)
@test convert(Some{Null}, Some(null)) === Some(null)
@test convert(Some, Some(nothing)) === Some(nothing)
@test convert(Some{Void}, Some(nothing)) === Some(nothing)

@test convert(Union{Some, Null}, null) === null
@test convert(Union{Some, Null}, null) === null
@test convert(Union{Some{Int}, Null}, null) === null

@test_throws MethodError convert(Some, nothing)
@test_throws MethodError convert(Some{Int}, nothing)
@test_throws MethodError convert(Union{Some, Null}, nothing)
@test_throws MethodError convert(Union{Some{Int}, Null}, nothing)

## show()

@test sprint(show, null) == "null"
@test sprint(show, Some(1)) == "Some(1)"
@test sprint(showcompact, Some(1)) == "1"
@test sprint(show, Some(Some(1))) == "Some(Some(1))"
@test sprint(showcompact, Some(Some(1))) == "1"

## isnull()

@test isnull(null) === true
@test isnull(nothing) === false
@test isnull(1) === false
@test isnull(Some(1)) === false
@test isnull(Some(null)) === false

## get()
@test get(Some(1)) === 1
@test get(Some(null)) === null
@test_throws NullException get(null)

@test get(Some(1), 0) === 1
@test get(Some(null), 0) === null
@test get(null, 0) === 0
