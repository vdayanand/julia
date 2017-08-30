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

function GitCredential(cfg::GitConfig, url::AbstractString)
    fill!(cfg, parse(GitCredential, url))
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

"""
    ismatch(url, git_cred) -> Bool

Checks if the `git_cred` is valid for the given `url`.
"""
function ismatch(url::AbstractString, git_cred::GitCredential)
    isempty(url) && return true

    m = match(URL_REGEX, url)
    m === nothing && error("Unable to parse URL")

    # Empty URL parts match everything
    m[:scheme] === nothing ? true : m[:scheme] == git_cred.protocol &&
    m[:host] === nothing ? true : m[:host] == git_cred.host &&
    m[:path] === nothing ? true : m[:path] == git_cred.path &&
    m[:user] === nothing ? true : m[:user] == git_cred.username
end

function isfilled(cred::GitCredential)
    !isempty(cred.username) && !isempty(cred.password)
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

function fill!(cfg::GitConfig, cred::GitCredential)
    username = Nullable{String}()
    for entry in GitConfigIter(cfg, r"credential.*")
        section, url, name, value = split(entry)

        # Only use configuration settings where the URL applies to the git credential
        ismatch(url, cred) || continue

        # https://git-scm.com/docs/gitcredentials#_configuration_options
        if name == "helper"
            helper = parse(GitCredentialHelper, value)
            fill!(helper, cred)
        elseif name == "username" && isnull(username)
            username = Nullable{String}(value)
        end

        # "Once Git has acquired both a username and a password, no more helpers will be
        # tried." – https://git-scm.com/docs/gitcredentials#gitcredentials-helper
        isfilled(cred) && break
    end

    # Default to the configuration username when one is not filled in
    if isempty(cred.username)
        cred.username = Base.get(username, "")
    end

    return cred
end

struct GitCredentialHelper
    cmd::Cmd
end

function Base.parse(::Type{GitCredentialHelper}, helper::AbstractString)
    # The helper string can take on different behaviors depending on the value:
    # - "Code after `!` evaluated in shell" – https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage
    # - "If the helper name is not an absolute path, then the string `git credential-` is
    #   prepended." – https://git-scm.com/docs/gitcredentials#gitcredentials-helper
    if startswith(helper, '!')
        cmd_str = helper[2:end]
    elseif isabspath(first(Base.shell_split(helper)))
        cmd_str = helper
    else
        cmd_str = "git credential-$helper"
    end

    GitCredentialHelper(`$(Base.shell_split(cmd_str)...)`)
end

function run!(helper::GitCredentialHelper, operation::AbstractString, cred::GitCredential)
    cmd = `$(helper.cmd) $operation`
    output, input, p = readandwrite(cmd)

    # Provide the helper with the credential information we know
    write(input, cred)
    write(input, "\n")
    close(input)

    # Process the response from the helper
    Base.read!(output, cred)
    close(output)

    return cred
end

function run(helper::GitCredentialHelper, operation::AbstractString, cred::GitCredential)
    run!(helper, operation, deepcopy(cred))
end

# The available actions between using `git credential` and helpers are slightly different.
# We will directly interact with the helpers as that way we can request credential
# information without a prompt (helper `get` vs. git credential `fill`).
# https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage

fill!(helper::GitCredentialHelper, cred::GitCredential) = run!(helper, "get", cred)
approve(helper::GitCredentialHelper, cred::GitCredential) = run(helper, "store", cred)
reject(helper::GitCredentialHelper, cred::GitCredential) = run(helper, "erase", cred)
