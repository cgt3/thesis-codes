using Colors
using ComponentArrays
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using RecursiveArrayTools: ArrayPartition
using SparseArrays, StaticArrays, StructArrays
using Triangulate
using TriplotBase, TriplotRecipes

using TriplotRecipes: TriPseudocolor
using PathIntersections
using StartUpDG

include("alive.jl")

## RHS functions:
include("srd_pacman_rhs.jl")

import ComponentArrays: ComponentArray
ComponentArray(x::NamedArrayPartition) = 
    ComponentArray(NamedTuple(Pair.(propertynames(x), getfield(x, :array_partition).x)))

## Boundary condition and forcing functions:
# Boundary conditions/forcing:
forcing(x, y, t) = 0*x
BOUNDARY_TOL = 1e-12
function BC(x,y,t, U)
    # if abs(x - domain.x_lb) < BOUNDARY_TOL || abs(x - domain.x_ub) < BOUNDARY_TOL ||
    #    abs(y - domain.y_lb) < BOUNDARY_TOL || abs(y - domain.y_ub) < BOUNDARY_TOL
    #     return 0.0, 0.0, 0.0
    # end

    
    if abs(x - domain.x_ub) < BOUNDARY_TOL || abs(y - domain.y_ub) < BOUNDARY_TOL
         # return 0.0, 0.0, 0.0
         return p_soln(x,y,t), v_soln(x,y,t)...
    end

    if abs(x - domain.x_lb) < BOUNDARY_TOL || abs(y - domain.y_lb) < BOUNDARY_TOL
         return p_soln(x,y,t), v_soln(x,y,t)...
     end

    r = hypot(x, y)
    theta = atan(y, x)
    if theta < 0
        theta += 2*pi
    end

    # Zero-vel: On the circular boundary or interior of the pacman
    if theta >= phi_top - BOUNDARY_TOL && theta <= phi_btm + BOUNDARY_TOL
        return U[1], 0.0, 0.0
    end

    return U
 end

## Simulation code:
N_pacman = 6
R_pacman = 1.0
phi_top = pi/N_pacman
phi_btm = (2*N_pacman-1)*pi/N_pacman
pacman_curve = PresetGeometries.Pacman(R=R_pacman, first_jaw=phi_top, second_jaw=phi_btm)
objects = (pacman_curve,)

domain = (; x_lb=-3.3, x_ub=3.0, y_lb=-3.3, y_ub=3.0)

t_end = 1
# N_deg = 1;
cells_per_dimension_x = 33
cells_per_dimension_y = 33
coordinates_min = (domain.x_lb, domain.y_lb)
coordinates_max = (domain.x_ub, domain.y_ub)


tspan = (0.0, t_end)
t_save = LinRange(tspan..., 100)
N_deg_all = [4]

L2_error = zeros(length(t_save), length(N_deg_all), 4)
inf_error = zeros(length(t_save), length(N_deg_all), 4)
l2_error = zeros(length(t_save), length(N_deg_all), 4)

L2_error_rel = zeros(length(t_save), length(N_deg_all), 4)
inf_error_rel = zeros(length(t_save), length(N_deg_all), 4)
l2_error_rel = zeros(length(t_save), length(N_deg_all), 4)

I_deg = 1;
# for I_deg in 1:length(N_deg_all)
    N_deg = N_deg_all[I_deg]

    rd = RefElemData(Quad(), N=N_deg)
    md = MeshData(rd, objects, cells_per_dimension_x, cells_per_dimension_y, coordinates_min=coordinates_min, coordinates_max=coordinates_max)
    state_redistr = StateRedistribution(rd, md)

    # Calculate the true solution using Lucas' code:
    x = md.x
    y = md.y
    include("pacman_solution_lucas.jl")


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

    # Bx, By = 10.0, 10.0
    # x0, y0 = 2.5, 0.0
    # pulse_mag = 2.0
    # gaussian(x,y) = @. exp(-Bx*(x - x0)^2 - By*(y-y0)^2)
    # Set the initial condition
    # p0 =  pulse_mag*gaussian(x, y)
    # v1_0 = 0*x
    # v2_0 = 0*x
    p0 = p_soln.(md.x, md.y, 0.0)
    v0 = v_soln.(md.x, md.y, 0.0)
    v1_0 = getindex.(v0, 1)
    v2_0 = getindex.(v0, 2)

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

    # tspan = (0.0, t_end)
    # t_save = LinRange(tspan..., 100)
    prob = ODEProblem(rhs_state_redistr!, u, tspan, params)
    print("Solving the ODE\n")

    sol = solve(prob, Tsit5(), dt=1e-4, save_everystep = false, saveat=t_save, callback=AliveCallback(alive_interval=50))

    #)
    n_t = length(sol.u)
    println("Finished solving.")

#     for k = 1:n_t
#         if  k % 50 == 0
#             println("k = $k")
#         end
#         # Calculate true solution
#         p_final = p_soln.(md.x, md.y, t_save[k])
#         v_final = v_soln.(md.x, md.y, t_save[k])
#         v1_final = getindex.(v_final, 1)
#         v2_final = getindex.(v_final, 2)

#         # Calculate the sup error
#         p_error = sol.u[n_t].x[1] - p_final
#         v1_error = sol.u[n_t].x[2] - v1_final
#         v2_error = sol.u[n_t].x[3] - v2_final

#         p_inf, v1_inf, v2_inf = norm(p_error, Inf), norm(v1_error, Inf), norm(v2_error, Inf)
#         total_inf = norm([p_inf v1_inf v2_inf], Inf)
        
#         p_inf_rel, v1_inf_rel, v2_inf_rel = p_inf/norm(p_final, Inf), v1_inf/norm(v1_final, Inf), v2_inf/norm(v2_final, Inf)
#         total_inf_rel = norm([p_inf_rel v1_inf_rel v2_inf_rel], Inf)

#         inf_error[k, I_deg, :] = [total_inf p_inf v1_inf v2_inf]
#         inf_error_rel[k, I_deg, :] = [total_inf_rel p_inf_rel v1_inf_rel v2_inf_rel]
        

#         p_l2, v1_l2, v2_l2 = norm(p_error), norm(v1_error), norm(v2_error)
#         total_l2 = norm([p_l2 v1_l2 v2_l2])
#         l2_error[k, I_deg, :] = [total_l2 p_l2 v1_l2 v2_l2]

#         p_l2_soln, v1_l2_soln, v2_l2_soln = norm(p_final), norm(v1_final), norm(v2_final)
#         p_l2_rel, v1_l2_rel, v2_l2_rel = p_l2/p_l2_soln, v1_l2/v1_l2_soln, v2_l2/v2_l2_soln
#         total_l2_rel = norm([p_l2 v1_l2 v2_l2]) / norm([p_l2_soln v1_l2_soln v2_l2_soln])
#         l2_error_rel[k, I_deg, :] = [total_l2_rel p_l2_rel v1_l2_rel v2_l2_rel]

#         # # L2: Cartesian cells
#         p_final_q = p_soln.(md.xq, md.yq, t_save[k])
#         v_final_q = v_soln.(md.xq, md.yq, t_save[k])
#         v1_final_q = getindex.(v_final_q, 1)
#         v2_final_q = getindex.(v_final_q, 2)

#         u_k = sol.u[k]
#         p_error  = similar(md.xq)
#         v1_error = similar(md.xq)
#         v2_error = similar(md.xq)
#         for e in 1:size(u_k.x[1].cartesian, 2)
#             p_error.cartesian[:,e]  = rd.Vq * u_k.x[1].cartesian[:,e] - p_final_q.cartesian[:,e]
#             v1_error.cartesian[:,e] = rd.Vq * u_k.x[2].cartesian[:,e] - v1_final_q.cartesian[:,e]
#             v2_error.cartesian[:,e] = rd.Vq * u_k.x[3].cartesian[:,e] - v2_final_q.cartesian[:,e]
#         end

#         # L2: Cut cells
#         for (e, elem) in enumerate(physical_frame_elements)
#             VDM = vandermonde(elem, rd.N, md.x.cut[:, e], md.y.cut[:, e])
#             Vq, _ = map(A -> A / VDM, basis(elem, rd.N, md.xq.cut[:,e], md.yq.cut[:, e]))

#             p_error.cut[:,e]  = Vq * u_k.x[1].cut[:,e] - p_final_q.cut[:,e]
#             v1_error.cut[:,e] = Vq * u_k.x[2].cut[:,e] - v1_final_q.cut[:,e]
#             v2_error.cut[:,e] = Vq * u_k.x[3].cut[:,e] - v2_final_q.cut[:,e]
#         end
#         p_soln_L2 = sqrt(sum(md.wJq .* p_final_q.^2))
#         v1_soln_L2 = sqrt(sum(md.wJq .* v1_final_q.^2))
#         v2_soln_L2 = sqrt(sum(md.wJq .* v2_final_q.^2))
#         total_soln_L2 = norm([p_soln_L2 v1_soln_L2 v2_soln_L2])

#         p_L2, v1_L2, v2_L2 = sqrt(sum(md.wJq .* p_error.^2)), sqrt(sum(md.wJq .* v1_error.^2)), sqrt(sum(md.wJq .* v2_error.^2))
#         total_L2 = norm([p_L2 v1_L2 v2_L2])

#         L2_error[k, I_deg, :] = [total_L2 p_L2 v1_L2 v2_L2]
#         L2_error_rel[k, I_deg, :] = [total_L2/total_soln_L2 p_L2/p_soln_L2 v1_L2/v1_soln_L2 v2_L2/v2_soln_L2]
#     end

#     println("L2 error for N_deg = ($N_deg, t_end) = $(l2_error[n_t, I_deg, 1])")
# end # N_deg

# println("Error for (nx, ny) = ($cells_per_dimension_x, $cells_per_dimension_y), N = $N_deg")
# println("Inf:\n $total_inf\n $p_inf\n $v1_inf\n $v2_inf")
# println("L2:\n  $total_L2\n $p_L2\n $v1_L2\n $v2_L2")

## Plotting ==============================================================================
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
plot_lim = 1.75
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
PLOT_SOLUTION = 1;
PLOT_TRUE_SOLUTION = 2;
PLOT_ERROR = 3;
PLOT_PTWS_ERROR = 4;

plot_type = PLOT_SOLUTION;
field = 1
n_t = length(sol.u)

# @gif for k in t_i
k = 100
    p_true = p_soln.(md.x, md.y, t_save[k])
    v_true = v_soln.(md.x, md.y, t_save[k])
    v1_true = getindex.(v_true, 1)
    v2_true = getindex.(v_true, 2)

    pt, v1_t, v2_t = ComponentArray.((p_true, v1_true, v2_true))    
    u_true = ArrayPartition(pt, v1_t, v2_t)

    plist = TriPseudocolor[]

    error_max = 0.0;
    # plist_cartesian = TriPseudocolor[]
    for e=1:size(md.x.cartesian,2)
        if plot_type == PLOT_SOLUTION
            u_e = sol.u[k].x[field].cartesian[:,e]
        elseif plot_type == PLOT_TRUE_SOLUTION
            u_e = u_true.x[field].cartesian[:,e]
        elseif plot_type == PLOT_ERROR
            u_e = sol.u[k].x[field].cartesian[:,e] - u_true.x[field].cartesian[:,e]
        elseif plot_type == PLOT_PTWS_ERROR
            u_e = sqrt.( (sol.u[k].x[1].cartesian[:,e] - u_true.x[1].cartesian[:,e]).^2 + 
                (sol.u[k].x[2].cartesian[:,e] - u_true.x[2].cartesian[:,e]).^2 + 
                (sol.u[k].x[3].cartesian[:,e] - u_true.x[3].cartesian[:,e]).^2
            )
        end


        u_ef = rd.Vf * u_e
        # push!(plist_cartesian, TriPseudocolor( vcat(md.xf.cartesian[:,e], md.x.cartesian[:,e]),
        push!(plist, TriPseudocolor( vcat(md.xf.cartesian[:,e], md.x.cartesian[:,e]),
                                     vcat(md.yf.cartesian[:,e], md.y.cartesian[:,e]),
                                     vcat(u_ef, u_e), t_cartesian ) )
    end
    # plist_cut = TriPseudocolor[]
    for e=1:size(md.x.cut,2)
        Vf = face_interpolation_matrices[e]

        if plot_type == PLOT_SOLUTION
            u_e = sol.u[k].x[field].cut[:,e]
        elseif plot_type == PLOT_TRUE_SOLUTION
            u_e = u_true.x[field].cut[:,e]
        elseif plot_type == PLOT_ERROR
            u_e = sol.u[k].x[field].cut[:,e] - u_true.x[field].cut[:,e]
        elseif plot_type == PLOT_PTWS_ERROR
            u_e = sqrt.( (sol.u[k].x[1].cut[:,e] - u_true.x[1].cut[:,e]).^2 + 
                (sol.u[k].x[2].cut[:,e] - u_true.x[2].cut[:,e]).^2 + 
                (sol.u[k].x[3].cut[:,e] - u_true.x[3].cut[:,e]).^2
            )
        end

        u_ef = Vf * u_e

        # push!(plist_cut, TriPseudocolor( vcat(md.xf.cut[cut_face_nodes[e]], md.x.cut[:,e]),
        push!(plist, TriPseudocolor( vcat(md.xf.cut[cut_face_nodes[e]], md.x.cut[:,e]),
                                     vcat(md.yf.cut[cut_face_nodes[e]], md.y.cut[:,e]),
                                     vcat(u_ef, u_e), t_cut[e] ) )
    end

  
    threshold = 2;
    neg = palette(:PuBu)
    pos = palette(:Purples)
    colors = vcat(neg[end:-1:1], pos[1:end])
    colorPos = range(0.0, 1.0, length(colors))

    colorGrad = cgrad(colors, colorPos);
    plot(plist, ylims=(domain.y_lb, domain.y_ub), 
        xlims=(domain.x_lb, domain.x_ub), 
        color=colorGrad, 
        clims=(-threshold, threshold), 
        colorbar=true,
        rightmargin=10Plots.mm)
    
    # plot(plist, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-plot_lim, plot_lim), colormap=:RdBu, colorbar=false)
    # plot(plist, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), colormap=:viridis, colorbar=false)

    # plot!(plist_objects, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-plot_lim, plot_lim), color=colormap("Grays"), colorbar=false)
    
    # plot!(plist_objects, ylims=(domain.y_lb, domain.y_ub), xlims=(domain.x_lb, domain.x_ub), clims=(-plot_lim, plot_lim), colormap=:viridis, colorbar=false)

# end

