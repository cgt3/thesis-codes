using ComponentArrays
using LinearAlgebra
using Plots
using RecursiveArrayTools: ArrayPartition
using SparseArrays, StaticArrays, StructArrays

using PathIntersections
using StartUpDG

## RHS functions:
include("srd_wave_rhs.jl")

## Simulation code:
delta=0.001
cells_per_dimension = 8
circle = PresetGeometries.Circle(R=0.7-delta, x0=0, y0=0)
rd = RefElemData(Quad(), N=4)
md = MeshData( rd, (circle, ), cells_per_dimension, cells_per_dimension)
state_redistr = StateRedistribution(rd, md)


(; physical_frame_elements, cut_face_nodes) = md.mesh_type
(; xf, yf, x, y) = md
face_interpolation_matrices = Matrix{Float64}[]
LIFT = Matrix{Float64}[]
Dx_skew, Dy_skew = Matrix{Float64}[], Matrix{Float64}[]
M_cut = Matrix{Float64}[]
for (e, elem) in enumerate(physical_frame_elements)

    VDM = vandermonde(elem, rd.N, x.cut[:, e], y.cut[:, e])
    Vq, Vrq, Vsq = map(A -> A / VDM, basis(elem, rd.N, md.xq.cut[:,e], md.yq.cut[:, e]))
   
    M  = Vq' * diagm(md.wJq.cut[:, e]) * Vq
    Qr = Vq' * diagm(md.wJq.cut[:, e]) * Vrq
    Qs = Vq' * diagm(md.wJq.cut[:, e]) * Vsq    
    Dx_skew_e, Dy_skew_e = M \ (0.5 * (Qr - Qr')), M \ (0.5 * (Qs - Qs'))
    
    Vf = vandermonde(elem, rd.N, xf.cut[cut_face_nodes[e]], yf.cut[cut_face_nodes[e]]) / VDM

    # don't include jacobian scaling in LIFT matrix (for consistency with the Cartesian mesh)
    md.mesh_type.cut_cell_data.wJf[cut_face_nodes[e]]
    w1D = reshape(rd.wf, :, rd.num_faces)[:,1] # extract first face weights
    w1D = 2 * w1D / sum(w1D) # normalize
    num_cut_faces = length(md.mesh_type.cut_face_nodes[e]) ÷ length(w1D)
    wf = repeat(w1D, 1, num_cut_faces)    

    push!(LIFT, M \ (Vf' * diagm(vec(wf))))
    push!(face_interpolation_matrices, Vf)
    push!(Dx_skew, Dx_skew_e)
    push!(Dy_skew, Dy_skew_e)
    push!(M_cut, M)
end

# Set the initial condition
x0 = 0.667
y0 = 0
beta = 100
p0 = @. exp(-beta * ((md.x-x0)^2 + (md.y-y0)^2))
v1_0 = @. 0 * md.x
v2_0 = @. 0 * md.x

u = ArrayPartition(p0, v1_0, v2_0)
du = similar(u)
fill!(du, 0.0)

forcing(x, y, t) = @. 0*x
BC(x,y,t, U) = @. 0.5*U[1], 0.5*U[2], 0.5*U[3]

function rhs_state_redistr!(du, u, params, t)
    @unpack rhs_params, use_srd, add_penalty = params

    # Apply state redistribution
    if use_srd == true
        for f in 1:length(u.x)
            state_redistr(u.x[f])
        end
    end

    # Update the RHS
    rhs!(du, u, rhs_params, t, add_penalty=add_penalty)
end

## Simulate the PDE
wJf = md.mesh_type.cut_cell_data.wJf
uf = ArrayPartition( similar(wJf), similar(wJf), similar(wJf))
cut_operators = (; LIFT, face_interpolation_matrices, Dx_skew, Dy_skew)
uP=similar(uf)

aux_mem = allocate_rhs_aux_memory(u, uf, rd)
rhs_params = (; cut_operators, md, rd, uf, uP, tau=1, forcing, BC, aux_mem)

P_params = (; rhs_params, use_srd=false, add_penalty=false)
PS_params = (; rhs_params, use_srd=true, add_penalty=false)
P_pen_params = (; rhs_params, use_srd=false, add_penalty=true)
PS_pen_params = (; rhs_params, use_srd=true, add_penalty=true)

M_cartesian = kron(md.J*I(StartUpDG.num_cartesian_elements(md)), rd.M)
M_global_block = Matrix(blockdiag(sparse(M_cartesian), sparse.(M_cut)...))
M_global = kron(I(3), M_global_block)

P = zeros(length(u), length(u))
PS = zeros(length(u), length(u))
P_pen = zeros(length(u), length(u))
PS_pen = zeros(length(u), length(u))

e_i = fill!(similar(u), zero(eltype(u)))
for i in 1:length(u)
    global e_i
    
    e_i[i] = 1
    rhs_state_redistr!(du, e_i, P_params, 0.0)
    P[:,i] .= du
    e_i .= 0

    e_i[i] = 1
    rhs_state_redistr!(du, e_i, PS_params, 0.0)
    PS[:,i] .= du
    e_i .= 0

    e_i[i] = 1
    rhs_state_redistr!(du, e_i, P_pen_params, 0.0)
    P_pen[:,i] .= du
    e_i .= 0

    e_i[i] = 1
    rhs_state_redistr!(du, e_i, PS_pen_params, 0.0)
    PS_pen[:,i] .= du
    e_i .= 0
end

P_eig = eigvals(P);
P_pen_eig = eigvals(P_pen);

PS_eig = eigvals(PS);
PS_pen_eig = eigvals(PS_pen);

P_max = maximum(abs.(P_eig));
PS_max = maximum(abs.(PS_eig));
P_pen_max =  maximum(abs.(P_pen_eig));
PS_pen_max =  maximum(abs.(PS_pen_eig));

ratio_noPen = P_max / PS_max;
ratio_pen = P_pen_max / PS_pen_max;

println("No SRD:")
println("  Max, no penalty: $P_max")
println("  Max, w  penalty: $P_pen_max")

println("With SRD:")
println("  Max, no penalty: $PS_max")
println("  Max, w  penalty: $PS_pen_max")

println("Ratios:")
println("  No penalty: $ratio_noPen")
println("  W  penalty: $ratio_pen")

plot_noSRD = scatter(P_eig, ratio=0.25, markersize=4, markerstrokewidth=0.2, xlims=(-1500,50), label="No SRD - No Penalty")
plot_noSRD = scatter!(P_pen_eig, ratio=0.25, markersize=4, markerstrokewidth=0.2, xlims=(-1500,50), label="No SRD - Penalty")
display(plot_noSRD);

plot_SRD = scatter(PS_eig, ratio=0.5, markersize=4, markerstrokewidth=0.2, xlims=(-200,5), label="SRD - No Penalty")
plot_SRD = scatter!(PS_pen_eig, ratio=0.5, markersize=4, markerstrokewidth=0.2, xlims=(-200,5), label="SRD - Penalty")
display(plot_SRD);