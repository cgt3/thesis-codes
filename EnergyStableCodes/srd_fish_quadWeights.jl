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
N_all = [1,2,3,4]
num_cut_cells=520

quadr_cond = zeros(4, num_cut_cells)
best_cond, worst_cond = zeros(4), zeros(4)
i_best, i_worst = zeros(Int32, 4), zeros(Int32, 4)

println("Fish Mesh:")
for N in N_all
    rd = RefElemData(Quad(), N=N)
    md = MeshData(rd, objects, cells_per_dimension, cells_per_dimension)
    state_redistr = StateRedistribution(rd, md)
    num_pts = size(md.wJq.cut,1)
    
    for i = 1:num_cut_cells
        quadr_cond[N,i] = sum(abs.(md.wJq.cut[:,i])) / sum(md.wJq.cut[:,i]) ;
    end

    worst_cond[N], i_worst[N] = maximum(quadr_cond[N,:]), argmax(quadr_cond[N,:])
    best_cond[N], i_best[N] = minimum(quadr_cond[N,:]), argmin(quadr_cond[N,:])

    println("  N=$N, num pts = $num_pts")
    println("    Best quadrature conditioning number:  $(best_cond[N])")
    println("    Worst quadrature conditioning number: $(worst_cond[N])")
end

N = N_all[end]
rd = RefElemData(Quad(), N=N)
md = MeshData(rd, objects, cells_per_dimension, cells_per_dimension)

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
p = plot(x_worst, y_worst, linewidth=4, lc=worst_color, label="Worst Conditioned Cut Element Quadrature" )
plot!(x_best, y_best,  linewidth=4, lc=best_color, label="Best Conditioned Cut Element Quadrature")

plot!(plist_objects, 
    ylims=(domain.y_lb, domain.y_ub), 
    xlims=(domain.x_lb, domain.x_ub), 
    axis=([], false), 
    clims=(-plot_lim, plot_lim), 
    color=colormap("Grays"), 
    colorbar=false)
png("quadrWts_fish_all")
display(p)

# From inspection, fish #1 has the worst conditioned element
i_fish = 1
x0 = -0.5
y0 = 0
dy = 0.125
dx = 0.2
zoomedIn = (;y_lb=y0-dy, y_ub=y0+dy, x_lb=x0-dx, x_ub=x0+dx)
mesh_lines = [domain.x_lb + i/cells_per_dimension*(domain.x_ub-domain.x_lb) for i=1:cells_per_dimension]

p = vline(mesh_lines, label=:none, lc=:lightgray)
hline!(mesh_lines, label=:none, lc=:lightgray)
plot!(plist_objects[i_fish],   
    ylims=(zoomedIn.y_lb, zoomedIn.y_ub), 
    xlims=(zoomedIn.x_lb, zoomedIn.x_ub), 
    clims=(-plot_lim, plot_lim), 
    color=colormap("Grays"), 
    aspect_ratio=1, 
    colorbar=false)

plot!(x_worst, y_worst, linewidth=4, lc=worst_color, label=:none, axis=([], false) )
png("quadrWts_fish_worst")
display(p)

# Fish #2 has the best conditioned element
i_fish = 2
x0 = -0.1
y0 = -0.2
dy = 0.125
dx = 0.2
zoomedIn = (;y_lb=y0-dy, y_ub=y0+dy, x_lb=x0-dx, x_ub=x0+dx)
mesh_lines = [domain.x_lb + i/cells_per_dimension*(domain.x_ub-domain.x_lb) for i=1:cells_per_dimension]

p = vline(mesh_lines, label=:none, lc=:lightgray)
hline!(mesh_lines, label=:none, lc=:lightgray)
plot!(plist_objects[i_fish],
    ylims=(zoomedIn.y_lb, zoomedIn.y_ub), 
    xlims=(zoomedIn.x_lb, zoomedIn.x_ub), 
    clims=(-plot_lim, plot_lim), 
    color=colormap("Grays"), 
    aspect_ratio=1, 
    colorbar=false)

plot!(x_best, y_best, linewidth=4, lc=best_color, label=:none, axis=([], false))
png("quadrWts_fish_best")
display(p)


cond_sorted = sort(quadr_cond[N,:])
num_outliers_6 = sum(cond_sorted .< 1/6)
num_outliers_1p5 = sum(cond_sorted .< 1/1.5)
p = histogram(cond_sorted, label=:none, title="Distribution of Cut Element Quadrature\n Conditioning on Fish Mesh" )
png("quadrWts_fish_hist")
display(p)

p = plot(cond_sorted,
    ylims=(0, 7), 
    seriestype=:scatter, 
    markerstrokewidth=0, 
    ylabel="Conditioning Number",
    label=:none)
png("quadrWts_fish_condNums")
display(p)

