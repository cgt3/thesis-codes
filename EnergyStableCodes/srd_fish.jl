using Colors
using ComponentArrays
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using RecursiveArrayTools: ArrayPartition
using SparseArrays, StaticArrays, StructArrays
using Triangulate
using TriplotBase, TriplotRecipes
#using Trixi: AliveCallback

using TriplotRecipes: TriPseudocolor
using PathIntersections
using StartUpDG

## RHS functions:
include("srd_wave_rhs.jl")

## Simulation code:
f_domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)
domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)
cells_per_dimension = 120

fish_x0 = -0.5
fish_dx = 0.4
fish_dy0 = 0.2
fish_dy = 0.4
fish_scale = 0.23
objects = [ PresetGeometries.Fish(scale=fish_scale, x0=fish_x0 + fish_dx*j, y0=-fish_dy0*j + fish_dy*i) for j=0:3 for i=0:j]


rd = RefElemData(Quad(), N=4)
md = MeshData(rd, objects, cells_per_dimension, cells_per_dimension)
state_redistr = StateRedistribution(rd, md)


(; physical_frame_elements, cut_face_nodes) = md.mesh_type
(; xf, yf, x, y) = md
face_interpolation_matrices = Matrix{Float64}[]
LIFT = Matrix{Float64}[]
Dx_skew, Dy_skew = Matrix{Float64}[], Matrix{Float64}[]
M_cut = Matrix{Float64}[]
for (e, elem) in enumerate(physical_frame_elements)

    VDM = vandermonde(elem, rd.N, x.cut[:, e], y.cut[:, e]) # TODO: should these be md.x, md.y?
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

# Boundary conditions/forcing:
# Have zero pressure boundary conditions on all boundaries except x=-1
# x = -1: Send a pressure wave at the intial times
t_wave = 0.05
BOUNDARY_TOL = 1e-12
function p_BC(x,y,t,p)
    if t < t_wave && abs(x - domain.x_lb) < BOUNDARY_TOL
        return 2
    end

    if abs(y - domain.y_lb) < BOUNDARY_TOL || abs(y - domain.y_ub) < BOUNDARY_TOL
        return p
    end
    return 0
end

function v1_BC(x,y,t, v1)
    # if abs(y - domain.y_lb) < BOUNDARY_TOL || abs(y - domain.y_ub) < BOUNDARY_TOL
    #      return 0.5*v1
    # end
    return v1
 end

function v2_BC(x,y,t, v2)
#    if abs(y - domain.y_lb) < BOUNDARY_TOL || abs(y - domain.y_ub) < BOUNDARY_TOL
#         return 0.5*v2
#    end
   return v2
end

forcing(x, y, t) = 0*x
BC(x,y,t,U) = p_BC(x,y,t,U[1]), v1_BC(x,y,t, U[2]), v2_BC(x,y,t, U[3])

# Set the initial condition
p0 = @. 0*x
v1_0 = @. 0*x
v2_0 = @. 0*x

import ComponentArrays: ComponentArray
ComponentArray(x::NamedArrayPartition) = 
    ComponentArray(NamedTuple(Pair.(propertynames(x), getfield(x, :array_partition).x)))

p0, v1_0, v2_0 = ComponentArray.((p0, v1_0, v2_0))    
u = ArrayPartition(p0, v1_0, v2_0)
du = similar(u)
fill!(du, 0.0)

function rhs_state_redistr!(du, u, params, t)
    @unpack mesh_params, use_srd, add_penalty = params

    # Apply state redistribution
    if use_srd == true
        for f in 1:length(u.x)
            state_redistr(u.x[f])
        end
    end

    # Update the RHS
    rhs!(du, u, mesh_params, t, add_penalty=add_penalty)
end

## Simulate the PDE
wJf = md.mesh_type.cut_cell_data.wJf
uf = ArrayPartition( similar(wJf), similar(wJf), similar(wJf))
cut_operators = (; LIFT, face_interpolation_matrices, Dx_skew, Dy_skew)
uP=similar(uf)

aux_mem = allocate_rhs_aux_memory(u, uf, rd)
rhs_params = (; cut_operators, md, rd, uf, uP, tau=1, forcing, BC, aux_mem)

params = (; rhs_params, use_srd=true, add_penalty=true)

t_end = 0.1 #6
tspan = (0.0, t_end)
prob = ODEProblem(rhs_state_redistr!, u, tspan, params)
print("Solving the ODE\n")

sol = solve(prob, Tsit5(), dt=1e-4, save_everystep = false, saveat=LinRange(tspan..., 100))
#callback=AliveCallback(alive_interval=50))
n_t = length(sol.u)
println("Finished solving.")


# Plotting ==============================================================================
# Build the triangulation for Cartesian elements
triin_cartesian=Triangulate.TriangulateIO()
triin_cartesian.pointlist = hcat( vcat(md.xf.cartesian[:,1], md.x.cartesian[:,1]),
                                  vcat(md.yf.cartesian[:,1], md.y.cartesian[:,1]) )'
triout_cartesian, _ = triangulate("Q", triin_cartesian)
t_cartesian = triout_cartesian.trianglelist

# # Build the triangulations for each cut cells
# t_cut = Matrix{Int32}[]
# for e=1:size(md.x.cut,2)
#     triin_cut=Triangulate.TriangulateIO()
#     triin_cut.pointlist = hcat( vcat(md.xf.cut[cut_face_nodes[e]], md.x.cut[:,e]),
#                                 vcat(md.yf.cut[cut_face_nodes[e]], md.y.cut[:,e]) )'
#     triout_cut, _ = triangulate("Q", triin_cut)
#     t_cut_e = triout_cut.trianglelist
#     push!(t_cut, t_cut_e)
# end

# Build the triangulations for the cut cells
t_cut = Matrix{Int32}[]
for e=1:size(md.x.cut,2)
triin_cut=Triangulate.TriangulateIO()
    # triin_cut.pointlist = hcat( vcat(md.xf.cut[cut_face_nodes[e]], md.x.cut[:,e]),
    #                             vcat(md.yf.cut[cut_face_nodes[e]], md.y.cut[:,e]) )'
    # # Set the boundary nodes
    # n_boundary = length(md.xf.cut[cut_face_nodes[e]])
    # triin_cut.segmentlist = zeros(Int32, (2, n_boundary));
    # triin_cut.segmentlist[1,:] = 1:n_boundary
    # triin_cut.segmentlist[2,:] = mod.(1:n_boundary, n_boundary) .+ 1
    # triout_cut, _ = triangulate("pQ", triin_cut)

    
    triin_cut=Triangulate.TriangulateIO()
    triin_cut.pointlist = hcat( vcat(md.xf.cut[cut_face_nodes[e]], md.x.cut[:,e]),
                                vcat(md.yf.cut[cut_face_nodes[e]], md.y.cut[:,e]) )'
    triout_cut, _ = triangulate("Q", triin_cut)

    t_cut_e = triout_cut.trianglelist
    push!(t_cut, t_cut_e)
end

# Build the triangulations for the embedded objects to plot over
# since the boundary nodes do not contain cell corners and can 
# consequently cause issues
field = 1
plot_lim = 1
obj_val = 2*plot_lim
plist_objects = TriPseudocolor[]
s = 0:0.01:1
for i=1:length(objects)
    pts = objects[i].(s)
    x_obj, y_obj = getindex.(pts,1), getindex.(pts,2)

    triin_cut=Triangulate.TriangulateIO()
    triin_cut.pointlist = hcat( x_obj, y_obj )'
    # Set the boundary nodes
    n_pts = length(x_obj)
    triin_cut.segmentlist = zeros(Int32, (2, n_pts));
    triin_cut.segmentlist[1,:] = 1:n_pts
    triin_cut.segmentlist[2,:] = mod.(1:n_pts, n_pts) .+ 1
    triout_cut, _ = triangulate("pQ", triin_cut)
    t_obj_i = triout_cut.trianglelist

    push!(plist_objects, TriPseudocolor( x_obj, y_obj, 
                                        obj_val*ones(length(x_obj)), t_obj_i) )

end


# Plot every element for each time step
@gif for k in 1:n_t
    plist = TriPseudocolor[]
    # plist_cartesian = TriPseudocolor[]
    for e=1:size(md.x.cartesian,2)
        u_e = sol.u[k].x[field].cartesian[:,e]
        u_ef = rd.Vf * sol.u[k].x[field].cartesian[:,e]
        # push!(plist_cartesian, TriPseudocolor( vcat(md.xf.cartesian[:,e], md.x.cartesian[:,e]),
        push!(plist, TriPseudocolor( vcat(md.xf.cartesian[:,e], md.x.cartesian[:,e]),
                                     vcat(md.yf.cartesian[:,e], md.y.cartesian[:,e]),
                                     vcat(u_ef, u_e), t_cartesian ) )
    end
    # plist_cut = TriPseudocolor[]
    for e=1:size(md.x.cut,2)
        Vf = face_interpolation_matrices[e]
        u_e = sol.u[k].x[field].cut[:,e]
        u_ef = Vf * u_e
        
        # push!(plist_cut, TriPseudocolor( vcat(md.xf.cut[cut_face_nodes[e]], md.x.cut[:,e]),
        push!(plist, TriPseudocolor( vcat(md.xf.cut[cut_face_nodes[e]], md.x.cut[:,e]),
                                     vcat(md.yf.cut[cut_face_nodes[e]], md.y.cut[:,e]),
                                     vcat(u_ef, u_e), t_cut[e] ) )
    end
    plot(plist, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-plot_lim, plot_lim), color=colormap("RdBu"), colorbar=false)
    # plot(plist_cut, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-plot_lim, plot_lim), color=colormap("RdBu"), colorbar=false)
    plot!(plist_objects, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-plot_lim, plot_lim), color=colormap("Grays"), colorbar=false)
    # plot!(plist_cartesian, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-plot_lim, plot_lim), color=colormap("RdBu"), colorbar=true)
end

