"""
    GitCredential

Git credential information used in communication with git credential helpers. The field are
named using the [input/output key specification](https://git-scm.com/docs/git-credential#IOFMT).
"""
mutable struct GitCredential
    protocol::String
    host::String
    path::String
    username::String
    password::String

    function GitCredential(
            protocol::AbstractString="",
            host::AbstractString="",
            path::AbstractString="",
            username::AbstractString="",
            password::AbstractString="")
        c = new(protocol, host, path, username, password)
        finalizer(c, securezero!)
        return c
    end
end

function securezero!(cred::GitCredential)
    securezero!(cred.protocol)
    securezero!(cred.host)
    securezero!(cred.path)
    securezero!(cred.username)
    securezero!(cred.password)
    return cred
end

function Base.:(==)(a::GitCredential, b::GitCredential)
    return a.protocol == b.protocol &&
           a.host == b.host &&
           a.path == b.path &&
           a.username == b.username &&
           a.password == b.password
end

function Base.parse(::Type{GitCredential}, url::AbstractString)
    m = match(URL_REGEX, url)
    m === nothing && error("Unable to parse URL")
    return GitCredential(
        m[:scheme] === nothing ? "" : m[:scheme],
        m[:host],
        m[:path] === nothing ? "" : m[:path],
        m[:user] === nothing ? "" : m[:user],
        m[:password] === nothing ? "" : m[:password],
    )
end

function Base.copy!(a::GitCredential, b::GitCredential)
    # Note: deepcopy calls avoid issues with securezero!
    a.protocol = deepcopy(b.protocol)
    a.host = deepcopy(b.host)
    a.path = deepcopy(b.path)
    a.username = deepcopy(b.username)
    a.password = deepcopy(b.password)
    return a
end

function Base.write(io::IO, cred::GitCredential)
    !isempty(cred.protocol) && println(io, "protocol=", cred.protocol)
    !isempty(cred.host) && println(io, "host=", cred.host)
    !isempty(cred.path) && println(io, "path=", cred.path)
    !isempty(cred.username) && println(io, "username=", cred.username)
    !isempty(cred.password) && println(io, "password=", cred.password)
    nothing
end

function Base.read!(io::IO, cred::GitCredential)
    # https://git-scm.com/docs/git-credential#IOFMT
    while !eof(io)
        key, value = split(readline(io), '=')

        if key == "url"
            # Any components which are missing from the URL will be set to empty
            # https://git-scm.com/docs/git-credential#git-credential-codeurlcode
            copy!(cred, parse(GitCredential, value))
        else
            setfield!(cred, Symbol(key), String(value))
        end
    end

    return cred
end
