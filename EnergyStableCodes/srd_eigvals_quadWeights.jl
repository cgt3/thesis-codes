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
domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)

delta=0.001
cells_per_dimension = 8
circle = PresetGeometries.Circle(R=0.7-delta, x0=0, y0=0)
objects = [circle]
N_all = [1,2,3,4]

num_cut_cells = 20 # Known from previous runs of the code
quadr_cond = zeros(4, num_cut_cells)
best_cond, worst_cond = zeros(4), zeros(4)
i_best, i_worst = zeros(Int32, 4), zeros(Int32, 4)

println("Circle Mesh:")
for N in N_all
    rd = RefElemData(Quad(), N=N)
    md = MeshData(rd, (circle, ), cells_per_dimension, cells_per_dimension)
    num_pts = size(md.wJq.cut,1)

    for i = 1:num_cut_cells
        quadr_cond[N,i] = sum(abs.(md.wJq.cut[:,i])) / sum(md.wJq.cut[:,i]);
    end

    worst_cond[N], i_worst[N] = maximum(quadr_cond[N, :]), argmax(quadr_cond[N,:])
    best_cond[N], i_best[N] = minimum(quadr_cond[N,:]), argmin(quadr_cond[N,:])

    println("  N=$N, num_pts = $num_pts:")
    println("    Best quadrature conditioning number:  $(best_cond[N])")
    println("    Worst quadrature conditioning number: $(worst_cond[N])")
end


N = N_all[end]
rd = RefElemData(Quad(), N=N)
md = MeshData(rd, (circle, ), cells_per_dimension, cells_per_dimension)

worst_cell_boundary = md.mesh_type.cut_cell_data.cutcells[i_worst[N]]
best_cell_boundary = md.mesh_type.cut_cell_data.cutcells[i_best[N]]

s = 0:0.001:1
pts_worst = worst_cell_boundary.(s);
pts_best = best_cell_boundary.(s);
x_worst, y_worst = getindex.(pts_worst,1), getindex.(pts_worst,2)
x_best, y_best = getindex.(pts_best,1), getindex.(pts_best,2)

# Build the triangulations for the embedded objects to plot over
# since the boundary nodes do not contain cell corners and can 
# consequently cause issues
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

worst_color = "red"
best_color = "green4"
mesh_lines = [domain.x_lb + i/cells_per_dimension*(domain.x_ub-domain.x_lb) for i=1:cells_per_dimension]
p = vline(mesh_lines, label=:none, lc=:lightgray)
hline!(mesh_lines, label=:none, lc=:lightgray)
plot!(plist_objects, 
    ylims=(-0.95, -0.3), 
    xlims=(-0.7, -0.1), 
    aspect_ratio=1,
    axis=([], false),
    legend=:top,
    clims=(-plot_lim, plot_lim), 
    color=colormap("Grays"), 
    colorbar=false)

plot!(x_worst, y_worst, linewidth=4, lc=worst_color, label=:none )
plot!(x_best, y_best, linewidth=4, lc=best_color, label=:none)

png("quadrWts_circle_all")
display(p)

p = plot(N_all, best_cond, 
    seriestype=:scatter,
    ylims=(0, 1), 
    markerstrokewidth=0, 
    markersize=6,
    markershape=:utriangle,
    label="Best", 
    title="Best and Worst Cut Element Quadrature \n Conditioning Numbers")

plot!(N_all, worst_cond, 
    seriestype=:scatter,
    markerstrokewidth=0, 
    markersize=6,
    markershape=:utriangle,
    label="Worst")

png("quadrWts_circle_bestWorst")
display(p)


