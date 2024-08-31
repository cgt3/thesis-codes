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
domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)
f_domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)
delta=0.01
cells_per_dimension = 16
circle1 = PresetGeometries.Circle(R=0.3, x0=-0.5)
circle2 = PresetGeometries.Circle(R=0.3, x0=0.5)
rect = PresetGeometries.Rectangle(Lx=0.1, Ly=5 , x0=1)
objects = (circle1,)
rd = RefElemData(Quad(), N=3)
md = MeshData(rd, objects, cells_per_dimension, cells_per_dimension)
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

# MMS:
# p = cos(2*pi*t)*sin(pi*x)*sin(pi*y)
# => v1 = -0.5*sin(2*pi*t)*cos(pi*x)*sin(pi*x)
# => v2 = -0.5*sin(2*pi*t)*sin(pi*x)*cos(pi*y)
p_exact(x,y,t) = cos(2*pi*t)*sin(pi*x)*sin(pi*y)
v1_exact(x,y,t) = -0.5*sin(2*pi*t)*cos(pi*x)*sin(pi*y)
v2_exact(x,y,t) = -0.5*sin(2*pi*t)*sin(pi*x)*cos(pi*y)

forcing(x, y, t) = @. -pi*sin(2*pi*t)*sin(pi*x)*sin(pi*y) 
BC(x,y,t, U) = @. p_exact(x,y,t), v1_exact(x,y,t), v2_exact(x,y,t)

# Set the initial condition
p0 = @. p_exact(md.x, md.y, 0)
v1_0 = @. v1_exact(md.x, md.y, 0)
v2_0 = @. v2_exact(md.x, md.y, 0)

import ComponentArrays: ComponentArray
ComponentArray(x::NamedArrayPartition) = 
    ComponentArray(NamedTuple(Pair.(propertynames(x), getfield(x, :array_partition).x)))
p0, v1_0, v2_0 = ComponentArray.((p0, v1_0, v2_0)) 
u = ArrayPartition(p0, v1_0, v2_0)
du = similar(u)
fill!(du, 0.0)

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

 params = (; rhs_params, use_srd=true, add_penalty=true)

t_end = 1
tspan = (0.0, t_end)
prob = ODEProblem(rhs_state_redistr!, u, tspan, params)
print("Solving the ODE\n")

sol = solve(prob, Tsit5(), dt=1e-4, save_everystep = false, saveat=LinRange(tspan..., 100))
#callback=AliveCallback(alive_interval=50))
n_t = length(sol.u)
println("Finished solving.")


# #For generating denser plotting nodes ----------------------------------------
# #Gather all the x and y plotting points
# x_all = vec(rd.Vp * x.cartesian)
# y_all = vec(rd.Vp * y.cartesian)
# Vp = Matrix{Float64}[]

# u_plot = similar(x_all)
# i_cartesian = length(x_all)
# for e in 1:size(md.x.cut,2)
#     global x_all, y_all, u_plot
#     elem = md.mesh_type.physical_frame_elements[e]
#     x_cut, y_cut = StartUpDG.generate_sampling_points(md.mesh_type.curves, elem, rd, 100)
#     x_all = vcat(x_all, vec(md.x.cut[:,e]))
#     y_all = vcat(y_all, vec(md.y.cut[:,e]))

#     local VDM = vandermonde(elem, rd.N, x.cut[:, e], y.cut[:, e])
#     Vp_e = vandermonde(elem, rd.N, x_cut, y_cut) / VDM

#     push!(Vp, Vp_e)
#     u_plot = vcat(u_plot, vec(Vp_e * sol.u[1].x[field].cut[:,e]))
#     # u_plot = vcat(u_plot, vec(Vp_e * u.x[field].cut[:,e]))
# end

# field = 2;
# n_cartesian = length(sol.u[1].x[field].cartesian)
# i_start = n_cartesian + 1
# @gif for k in 1:n_t
#     u_plot[1:n_cartesian] .= sol.u[k].x[field].cartesian
#     for e in 1:size(md.x.cut,2)
#         i_end = i_start + length(sol.u[1].x[field].cut[:,e]) - 1
#         u_plot[i_start:i_end] .= vec(Vp[e] * sol.u[1].x[field].cut[:,e])
#         i_start = i_end + 1
#     end
#     scatter(x_all, y_all, u_plot, clims=(-lim, lim) )
#     # plot(plist)
# end

# # -----------------------------------------------------------------------------

# Plot over the embedded objects
num_obj_pts = 100
s = LinRange(0, 1, num_obj_pts)
obj_pts = zeros(num_obj_pts*length(objects), 2)
for j in eachindex(objects)
    for i in eachindex(s)
        pt = objects[j](s[i])
        obj_pts[num_obj_pts*(j-1)+i,:] .= pt
    end
end

# Build up the triangulation now that x and y are built up for all elements
lim = 1
triin_soln=Triangulate.TriangulateIO()
triin_soln.pointlist = hcat(vec(md.x), vec(md.y))'
triout_soln, _ = triangulate("Q", triin_soln)
t_soln = triout_soln.trianglelist

triin_objs=Triangulate.TriangulateIO()
triin_objs.pointlist = obj_pts'
triout_objs, _ = triangulate("Q", triin_objs)
t_obj = triout_objs.trianglelist
z_obj = -2*lim*ones(size(obj_pts,1))


field = 1;
@gif for k in 1:1#n_t
    plist = [TriPseudocolor(md.x,md.y,sol.u[k].x[field],t_soln), 
            TriPseudocolor(obj_pts[:,1],obj_pts[:,2],z_obj,t_obj)]
    plot(plist, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-lim, lim) )
    # plot(plist)
end

