module AcousticPACMAN
using SpecialFunctions

export pacman, pressure, axialvelocity, radialvelocity,
    surfacevibration, planewave

δ(m, n) = m == n
δ(m) = m == 0 ? 1 : 2

# Here we use a recurrence relationship to compute the derivative of the
# various Bessel functions, see <https://dlmf.nist.gov/10.6#E1> and
# <https://dlmf.nist.gov/10.6#E3>.
for C in (:besselj, :bessely, :hankelh1, :hankelh2)
    Cprime = Symbol(C, "prime")
    @eval begin
        function $Cprime(ν, z)
            if ν == 0
                return -$C(one(ν), z)
            else
                return ($C(ν - 1, z) - $C(ν + 1, z)) / 2
            end
        end
    end
end

function SpecialFunctions.besselh(nu::Int, k::Int, x::BigFloat)
    return besselh(BigFloat(nu), k, x)
end

function SpecialFunctions.besselh(nu::BigFloat, k::Int, x::BigFloat)
    if k == 1
        return complex(besselj(nu, x), bessely(nu, x))
    elseif k == 2
        return complex(besselj(nu, x), -bessely(nu, x))
    else
        throw(SpecialFunctions.AmosException(1))
    end
end

# These are functions (B.1) and (B.2) from
# <https://doi.org/10.1016/j.jcp.2017.06.018> respectively
# ϕ₀(N) = π / N
# Isin(ϕ₀, x, y) = (sinc((x-y)*ϕ₀/π) - sinc((x+y)*ϕ₀/π)) / 2
# Icos(ϕ₀, x, y) = (sinc((x+y)*ϕ₀/π) + sinc((x-y)*ϕ₀/π)) / 2
#
Isin(Ninv, x, y) = (sinc((x - y) * Ninv) - sinc((x + y) * Ninv)) / 2
Icos(Ninv, x, y) = (sinc((x + y) * Ninv) + sinc((x - y) * Ninv)) / 2

abstract type Fun end

pacman(f::Fun) = f.pac

struct PressureFun{P} <: Fun
    pac::P
end

struct AxialVelocityFun{P} <: Fun
    pac::P
end

struct RadialVelocityFun{P} <: Fun
    pac::P
end

struct Pacman{I,T,C<:AbstractArray{Complex{T}}}
    aS::C
    aA::C
    bS::C
    bA::C
    M::I
    N::I
    k::T
    r₀::T
    Z₀::T
end

abstract type InitialCondition end

struct InitialConditionCoefficients{A}
    VSvib::A
    VAvib::A
    PSinc::A
    PAinc::A
    VSinc::A
    VAinc::A
end

symmetricvibration(icc::InitialConditionCoefficients) = icc.VSvib
antisymmetricvibration(icc::InitialConditionCoefficients) = icc.VAvib
symmetricincidentpressure(icc::InitialConditionCoefficients) = icc.PSinc
antisymmetricincidentpressure(icc::InitialConditionCoefficients) = icc.PAinc
symmetricincidentvelocity(icc::InitialConditionCoefficients) = icc.VSinc
antisymmetricincidentvelocity(icc::InitialConditionCoefficients) = icc.VAinc

wavenumber(pac::Pacman) = pac.k
truncationorder(pac::Pacman) = pac.M
wedgenumber(pac::Pacman) = pac.N
radius(pac::Pacman) = pac.r₀
characteristicimpedance(pac::Pacman) = pac.Z₀
outersymmetricmodeamplitudes(pac::Pacman) = pac.aS
outerantisymmetricmodeamplitudes(pac::Pacman) = pac.aA
innersymmetricmodeamplitudes(pac::Pacman) = pac.bS
innerantisymmetricmodeamplitudes(pac::Pacman) = pac.bA
pressure(pac::Pacman) = PressureFun(pac)
axialvelocity(pac::Pacman) = AxialVelocityFun(pac)
radialvelocity(pac::Pacman) = RadialVelocityFun(pac)

function pacman(M, N, k, r₀, Z₀, ic::InitialCondition)
    AS, AA, rhsS, rhsA = getsystem(M, N, k, r₀, Z₀, ic)

    aS = AS \ rhsS

    # The zero antisymmetric mode doesn't contribute to
    # the solution.  So we just set it to zero and solve
    # for the other modes.
    aA = similar(aS)
    aA[1] = 0
    @views aA[2:end] .= AA[2:end, 2:end] \ rhsA[2:end]

    bS, bA = getinteriormodes(aS, aA, M, N, k * r₀, ic)

    return Pacman(aS, aA, bS, bA, M, N, k, r₀, Z₀)
end

function getinteriormodes(aS, aA, M, N, kr₀, ic::InitialCondition)
    T = typeof(kr₀)
    Ninv = one(T) / N

    K = div(M + 1, N) - 1

    icc = coefficients(ic, kr₀)
    psinc = symmetricincidentpressure(icc)
    painc = antisymmetricincidentpressure(icc)

    PSinc = zeros(eltype(psinc), M + 1)
    PAinc = zeros(eltype(painc), M + 1)

    PSinc[1:length(psinc)] .= psinc
    PAinc[1:length(painc)] .= painc

    bS = similar(aS, K + 1)
    bA = similar(aA, K + 1)

    # Precompute special functions for speed
    h = zeros(complex(T), M + 1)
    icosN = zeros(T, M + 1, M + 1)
    isinN = zeros(T, M + 1, M + 1)

    Threads.@threads :static for n = 0:M
        h[n + 1] = hankelh2(n, kr₀)
        for m = 0:M
            icosN[m + 1, n + 1] = Icos(Ninv, m, n * N)
            isinN[m + 1, n + 1] = Isin(Ninv, m, n * N + div(N, 2))
        end
    end

    Threads.@threads :static for η = 0:K
        bS[η + 1] =
            δ(η * N) * sum(@views (aS .* h .+ PSinc) .* icosN[:, η + 1]) /
            besselj(η * N, kr₀)
        bA[η + 1] =
            2 * sum(@views (aA .* h .+ PAinc) .* isinN[:, η + 1]) /
            besselj(η * N + div(N, 2), kr₀)
    end

    return (bS, bA)
end

function getsystem(M, N, k, r₀, Z₀, ic::InitialCondition)
    T = typeof(r₀)
    kr₀ = k * r₀
    Ninv = one(T) / N
    K = div(M + 1, N) - 1

    icc = coefficients(ic, kr₀)

    VSvib = symmetricvibration(icc)
    VAvib = antisymmetricvibration(icc)
    PSinc = symmetricincidentpressure(icc)
    PAinc = antisymmetricincidentpressure(icc)
    VSinc = symmetricincidentvelocity(icc)
    VAinc = antisymmetricincidentvelocity(icc)

    # Precompute special functions for speed
    h = zeros(complex(T), M + 1)
    hp = zeros(complex(T), M + 1)
    JcoefS = zeros(T, M + 1)
    JcoefA = zeros(T, M + 1)
    icosN = zeros(T, M + 1, M + 1)
    isinN = zeros(T, M + 1, M + 1)

    Threads.@threads :static for n = 0:M
        h[n + 1] = hankelh2(n, kr₀)
        hp[n + 1] = hankelh2prime(n, kr₀)
        JcoefS[n + 1] = Ninv * δ(n * N) * besseljprime(n * N, kr₀) / besselj(n * N, kr₀)
        JcoefA[n + 1] =
            Ninv * 2 * besseljprime(n * N + div(N, 2), kr₀) /
            besselj(n * N + div(N, 2), kr₀)
        for m = 0:M
            icosN[m + 1, n + 1] = Icos(Ninv, m, n * N)
            isinN[m + 1, n + 1] = Isin(Ninv, m, n * N + div(N, 2))
        end
    end

    AS = zeros(complex(T), M + 1, M + 1)
    AA = zeros(complex(T), M + 1, M + 1)
    rhsS = zeros(complex(T), M + 1)
    rhsA = zeros(complex(T), M + 1)

    Threads.@threads :static for n = 0:M
        # Build the system for the symmetric modes
        for i = 0:M
            J = zero(T)
            for η = 0:K
                J += JcoefS[η + 1] * icosN[n + 1, η + 1] * icosN[i + 1, η + 1]
            end
            AS[i + 1, n + 1] = δ(n, i) * hp[i + 1] / δ(i) - J * h[n + 1]
        end

        for nstar = 0:(length(PSinc) - 1)
            for η = 0:K
                rhsS[n + 1] +=
                    PSinc[nstar + 1] *
                    JcoefS[η + 1] *
                    icosN[nstar + 1, η + 1] *
                    icosN[n + 1, η + 1]
            end
        end

        if n < length(VSinc)
            rhsS[n + 1] += -VSinc[n + 1] / δ(n)
        end

        if n < length(VSvib)
            rhsS[n + 1] += -im * Z₀ * VSvib[n + 1] / δ(n)
        end

        for nstar = 0:(length(VSvib) - 1)
            rhsS[n + 1] += im * Ninv * Z₀ * VSvib[nstar + 1] * Icos(Ninv, n, nstar)
        end

        # Build the system for the antisymmetric modes
        for i = 0:M
            J = zero(T)
            for η = 0:K
                J += JcoefA[η + 1] * isinN[n + 1, η + 1] * isinN[i + 1, η + 1]
            end
            AA[i + 1, n + 1] = (δ(n, i) - δ(i, 0)) * hp[i + 1] / 2 - J * h[n + 1]
        end

        for nstar = 0:(length(PAinc) - 1)
            for η = 0:K
                rhsA[n + 1] +=
                    PAinc[nstar + 1] *
                    JcoefA[η + 1] *
                    isinN[nstar + 1, η + 1] *
                    isinN[n + 1, η + 1]
            end
        end

        if n < length(VAinc)
            rhsA[n + 1] += -VAinc[n + 1] / 2
        end

        if n < length(VAvib)
            rhsA[n + 1] += -im * Z₀ * (1 - δ(n, 0)) * VAvib[n + 1] / 2
        end

        for nstar = 0:(length(VAvib) - 1)
            rhsA[n + 1] += im * Ninv * Z₀ * VAvib[nstar + 1] * Isin(Ninv, n, nstar)
        end
    end

    return (AS, AA, rhsS, rhsA)
end

(p::PressureFun)(r, ϕ) = apply(p, r, ϕ)

@inline function apply(p::PressureFun, r, ϕ)
    pac = pacman(p)

    k = wavenumber(pac)
    T = eltype(k)
    r₀ = radius(pac)
    N = wedgenumber(pac)

    ϕ₀ = π / T(N)
    kr = k * r

    if r ≥ r₀
        # outside the pacman
        aS = outersymmetricmodeamplitudes(pac)
        aA = outerantisymmetricmodeamplitudes(pac)
        n = 0:(length(aS) - 1)
        val = sum(@. hankelh2(n, kr) * (aA * sin(n * ϕ) + aS * cos(n * ϕ)))
    elseif r < r₀ && -ϕ₀ ≤ ϕ ≤ ϕ₀
        # inside the mouth
        bS = innersymmetricmodeamplitudes(pac)
        bA = innerantisymmetricmodeamplitudes(pac)
        n = 0:(length(bS) - 1)
        val = sum(
            @. besselj(n * N + div(N, 2), kr) * bA * sin((n * N + div(N, 2)) * ϕ) +
               besselj(n * N, kr) * bS * cos(n * N * ϕ)
        )
    else
        val = zero(Complex{T})
    end

    return val
end

(p::AxialVelocityFun)(r, ϕ) = apply(p, r, ϕ)

@inline function apply(p::AxialVelocityFun, r, ϕ)
    pac = pacman(p)

    k = wavenumber(pac)
    T = eltype(k)
    r₀ = radius(pac)
    N = wedgenumber(pac)
    Z₀ = characteristicimpedance(pac)

    ϕ₀ = π / T(N)
    kr = k * r

    if r ≥ r₀
        # outside the pacman
        aS = outersymmetricmodeamplitudes(pac)
        aA = outerantisymmetricmodeamplitudes(pac)
        n = 0:(length(aS) - 1)
        val = im * sum(@. hankelh2(n, kr) * (aA * n * cos(n * ϕ) - aS * n * sin(n * ϕ))) / (kr * Z₀)
    elseif r < r₀ && -ϕ₀ ≤ ϕ ≤ ϕ₀
        # inside the mouth
        bS = innersymmetricmodeamplitudes(pac)
        bA = innerantisymmetricmodeamplitudes(pac)
        n = 0:(length(bS) - 1)
        val = im * sum(
            @. besselj(n * N + div(N, 2), kr) * bA * (n * N + div(N, 2)) * cos((n * N + div(N, 2)) * ϕ) -
               besselj(n * N, kr) * bS * n * N * sin(n * N * ϕ)
        ) / (kr * Z₀)
    else
        val = zero(Complex{T})
    end

    return val
end

(v::RadialVelocityFun)(r, ϕ) = apply(v, r, ϕ)

@inline function apply(v::RadialVelocityFun, r, ϕ)
    pac = pacman(v)

    k = wavenumber(pac)
    T = eltype(k)
    r₀ = radius(pac)
    N = wedgenumber(pac)
    Z₀ = characteristicimpedance(pac)

    ϕ₀ = π / T(N)
    kr = k * r

    if r ≥ r₀
        # outside the pacman
        aS = outersymmetricmodeamplitudes(pac)
        aA = outerantisymmetricmodeamplitudes(pac)
        n = 0:(length(aS) - 1)
        val = im * sum(@. hankelh2prime(n, kr) * (aA * sin(n * ϕ) + aS * cos(n * ϕ))) / Z₀
    elseif r < r₀ && -ϕ₀ ≤ ϕ ≤ ϕ₀
        # inside the mouth
        bS = innersymmetricmodeamplitudes(pac)
        bA = innerantisymmetricmodeamplitudes(pac)
        n = 0:(length(bS) - 1)
        val =
            im * sum(
                @. besseljprime(n * N + div(N, 2), kr) * bA * sin((n * N + div(N, 2)) * ϕ) +
                   besseljprime(n * N, kr) * bS * cos(n * N * ϕ)
            ) / Z₀
    else
        val = zero(Complex{T})
    end

    return val
end

struct SurfaceVibration{A} <: InitialCondition
    VSvib::A
    VAvib::A
end

surfacevibration(VSvib, VAvib) = SurfaceVibration(VSvib, VAvib)

function coefficients(sv::SurfaceVibration, kr)
    PSinc = similar(sv.VSvib, 0)
    PAinc = similar(sv.VSvib, 0)
    VSinc = similar(sv.VSvib, 0)
    VAinc = similar(sv.VSvib, 0)
    return InitialConditionCoefficients(sv.VSvib, sv.VAvib, PSinc, PAinc, VSinc, VAinc)
end

struct PlaneWave{T,I} <: InitialCondition
    M::I
    k::T
    ϕₛ::T
    Z₀::T
end

planewave(M, k, ϕₛ, Z₀) = PlaneWave(M, k, ϕₛ, Z₀)
order(pw::PlaneWave) = pw.M
wavenumber(pw::PlaneWave) = pw.k
incidentangle(pw::PlaneWave) = pw.ϕₛ
characteristicimpedance(pw::PlaneWave) = pw.Z₀

struct PlaneWavePressureFun{I} <: Fun
    pw::I
end
planewave(p::PlaneWavePressureFun) = p.pw

pressure(pw::PlaneWave) = PlaneWavePressureFun(pw)
(p::PlaneWavePressureFun)(r, ϕ) = apply(p, r, ϕ)
@inline function apply(p::PlaneWavePressureFun, r, ϕ)
    pw = planewave(p)
    k = wavenumber(pw)
    ϕₛ = incidentangle(pw)

    return cos(k * r * cos(ϕ - ϕₛ)) + im * sin(k * r * cos(ϕ - ϕₛ))
end

struct PlaneWaveRadialVelocityFun{I} <: Fun
    pw::I
end
planewave(p::PlaneWaveRadialVelocityFun) = p.pw

struct PlaneWaveAxialVelocityFun{I} <: Fun
    pw::I
end
planewave(p::PlaneWaveAxialVelocityFun) = p.pw

axialvelocity(pw::PlaneWave) = PlaneWaveAxialVelocityFun(pw)
(p::PlaneWaveAxialVelocityFun)(r, ϕ) = apply(p, r, ϕ)
@inline function apply(p::PlaneWaveAxialVelocityFun, r, ϕ)
    pw = planewave(p)
    k = wavenumber(pw)
    ϕₛ = incidentangle(pw)
    Z₀ = characteristicimpedance(pw)

    kr = k * r

    cϕ = cos(ϕ - ϕₛ)

    sϕ = sin(ϕ - ϕₛ)

    dpdϕ = kr * sϕ * sin(k * r * cϕ) - im * kr * sϕ * cos(k * r * cϕ)

    return im * dpdϕ / (kr * Z₀)
end

radialvelocity(pw::PlaneWave) = PlaneWaveRadialVelocityFun(pw)
(p::PlaneWaveRadialVelocityFun)(r, ϕ) = apply(p, r, ϕ)
@inline function apply(p::PlaneWaveRadialVelocityFun, r, ϕ)
    pw = planewave(p)
    k = wavenumber(pw)
    ϕₛ = incidentangle(pw)
    Z₀ = characteristicimpedance(pw)

    cϕ = cos(ϕ - ϕₛ)

    dpdr = -k * cϕ * sin(k * r * cϕ) + im * k * cϕ * cos(k * r * cϕ)

    return im * dpdr / (k * Z₀)
end

function coefficients(pw::PlaneWave{T}, kr) where {T}
    M = order(pw)
    ϕₛ = incidentangle(pw)

    Pinc = [δ(n) * im^n * besselj(T(n), kr) for n = 0:M]
    Vinc = [δ(n) * im^n * besseljprime(T(n), kr) for n = 0:M]
    S = [cos(n * ϕₛ) for n = 0:M]
    A = [sin(n * ϕₛ) for n = 0:M]

    PSinc = Pinc .* S
    PAinc = Pinc .* A
    VSinc = Vinc .* S
    VAinc = Vinc .* A
    VSvib = similar(PSinc, 0)
    VAvib = similar(PSinc, 0)

    return InitialConditionCoefficients(VSvib, VAvib, PSinc, PAinc, VSinc, VAinc)
end

end # module AcousticPACMAN
