using ComponentArrays
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Polyester
using RecursiveArrayTools: ArrayPartition
using SparseArrays, StaticArrays, StructArrays

using PathIntersections
using StartUpDG


function rhs!(du, u, params, t; add_penalty=false)  
    @unpack md, rd, cut_operators, uf, uP, forcing, BC, aux_mem = params

    @unpack face_interpolation_matrices = cut_operators
    @unpack cart_mem, cut_mem, uBC = aux_mem

    # interpolate to faces (3 allocations)
    uf.x[1].cartesian .= rd.Vf * u.x[1].cartesian
    uf.x[2].cartesian .= rd.Vf * u.x[2].cartesian
    uf.x[3].cartesian .= rd.Vf * u.x[3].cartesian

    num_cut_elements = size(u.x[1].cut,2)
    @batch per=thread for e in 1:num_cut_elements
        Vf = face_interpolation_matrices[e]
        uf.x[1].cut[ md.mesh_type.cut_face_nodes[e] ] .= Vf * u.x[1].cut[:,e] 
        uf.x[2].cut[ md.mesh_type.cut_face_nodes[e] ] .= Vf * u.x[2].cut[:,e] 
        uf.x[3].cut[ md.mesh_type.cut_face_nodes[e] ] .= Vf * u.x[3].cut[:,e] 
    end

    @batch for i in eachindex(uP.x)
        for id in eachindex(md.mapP)
            uP.x[i][id] = uf.x[i][md.mapP[id]]
        end
    end

    # Apply BCs
    @batch per=thread for i in eachindex(md.mapP)
        if i == md.mapP[i]
            uBC = BC(md.xf[i], md.yf[i], t, (uP.x[1][i], uP.x[2][i], uP.x[3][i]))
            uP.x[1][i] = 2*uBC[1] - uP.x[1][i]
            uP.x[2][i] = 2*uBC[2] - uP.x[2][i]
            uP.x[3][i] = 2*uBC[3] - uP.x[3][i] 
        end
    end

    du_cartesian = ArrayPartition( du.x[1].cartesian, du.x[2].cartesian, du.x[3].cartesian )
    u_cartesian  = ArrayPartition(  u.x[1].cartesian,  u.x[2].cartesian,  u.x[3].cartesian )
    uP_cartesian = ArrayPartition( uP.x[1].cartesian, uP.x[2].cartesian, uP.x[3].cartesian )
    uf_cartesian = ArrayPartition( uf.x[1].cartesian, uf.x[2].cartesian, uf.x[3].cartesian )
    rhs_cartesian!(du_cartesian, u_cartesian, uf_cartesian, uP_cartesian, (; tau=params.tau, md, rd, forcing, cart_mem), t, add_penalty=add_penalty)

    du_cut = ArrayPartition( du.x[1].cut, du.x[2].cut, du.x[3].cut )
    u_cut  = ArrayPartition(  u.x[1].cut,  u.x[2].cut,  u.x[3].cut )
    uf_cut = ArrayPartition( uf.x[1].cut, uf.x[2].cut, uf.x[3].cut )
    uP_cut = ArrayPartition( uP.x[1].cut, uP.x[2].cut, uP.x[3].cut )
    rhs_cut!(du_cut, u_cut, uf_cut, uP_cut, (;  tau=params.tau, md, cut_operators, forcing, cut_mem), t, add_penalty=add_penalty)
end

function rhs_cartesian!(du, u, uf, uP, params, t; add_penalty=false)
    @unpack tau, md, rd, forcing, cart_mem = params
    @unpack flux, dudr, duds, Dr, Ds = cart_mem

    @batch per=thread for i in eachindex(uf.x[1]) 
        p, v1, v2    = uf.x[1][i], uf.x[2][i], uf.x[3][i]
        pP, v1P, v2P = uP.x[1][i], uP.x[2][i], uP.x[3][i]

        flux.x[1][i] = v1P * md.nxJ.cartesian[i] + v2P * md.nyJ.cartesian[i]
        flux.x[2][i] = pP * md.nxJ.cartesian[i]
        flux.x[3][i] = pP * md.nyJ.cartesian[i]

        # Add in penalty terms
        if add_penalty
            nx = md.nxJ.cartesian[i] / md.Jf.cartesian[i]
            ny = md.nyJ.cartesian[i] / md.Jf.cartesian[i]
            vjump_normal = (v1P - v1) * nx + (v2P - v2) * ny
            flux.x[1][i] -= tau * (pP - p) * md.Jf.cartesian[i]
            flux.x[2][i] -= tau * vjump_normal * md.nxJ.cartesian[i]
            flux.x[3][i] -= tau * vjump_normal * md.nyJ.cartesian[i]
        end
    end
    du.x[1] .= 0.5 * rd.LIFT * flux.x[1]
    du.x[2] .= 0.5 * rd.LIFT * flux.x[2]
    du.x[3] .= 0.5 * rd.LIFT * flux.x[3]

    dudr.x[1] .= Dr * u.x[1]
    dudr.x[2] .= Dr * u.x[2]
    dudr.x[3] .= Dr * u.x[3]
    
    duds.x[1] .= Ds * u.x[1]
    duds.x[2] .= Ds * u.x[2]
    duds.x[3] .= Ds * u.x[3]

    @batch per=thread for e in 1:size(u.x[1], 2)
        for i in 1:size(u.x[1], 1)
            dpdr, dv1dr = dudr.x[1][i, e], dudr.x[2][i, e]
            dpds, dv2ds = duds.x[1][i, e], duds.x[3][i, e]

            dv1dx = md.rxJ.cartesian[e] * dv1dr
            dv2dy = md.syJ.cartesian[e] * dv2ds

            du_p = dv1dx + dv2dy
            du_v1 = md.rxJ.cartesian[e] * dpdr
            du_v2 = md.syJ.cartesian[e] * dpds

            du.x[1][i, e] += du_p
            du.x[2][i, e] += du_v1
            du.x[3][i, e] += du_v2
        end
    end

    du .= -du ./ md.J

    du.x[1] .+= forcing(md.x.cartesian, md.y.cartesian, t)
end

# uf = nodes on faces
# uP = ghost nodes on boundaries
function rhs_cut!(du, u, uf, uP, params, t;  add_penalty=false)
    @unpack tau, md, cut_operators, forcing, cut_mem = params
    @unpack flux = cut_mem
    @unpack LIFT, Dx_skew, Dy_skew = cut_operators
        
    fill!(du, zero(eltype(du)))

    # Calculate the flux terms for each surface node
    @batch per=thread for i in eachindex(uf.x[1])
        p, v1, v2    = uf.x[1][i], uf.x[2][i], uf.x[3][i]
        pP, v1P, v2P = uP.x[1][i], uP.x[2][i], uP.x[3][i]

        flux.x[1][i] = (v1P * md.nxJ.cut[i] + v2P * md.nyJ.cut[i])
        flux.x[2][i] = pP * md.nxJ.cut[i] 
        flux.x[3][i] = pP * md.nyJ.cut[i] 

        if add_penalty
            nx = md.nxJ.cut[i] / md.Jf.cut[i]
            ny = md.nyJ.cut[i] / md.Jf.cut[i]
            vjump_normal = (v1P - v1) * nx + (v2P - v2) * ny
            flux.x[1][i] -= tau * (pP-p) * md.Jf.cut[i]
            flux.x[2][i] -= tau * vjump_normal * md.nxJ.cut[i]
            flux.x[3][i] -= tau * vjump_normal * md.nyJ.cut[i]
        end
    end

    # Combine the surface and volume terms
    @batch per=thread for e in 1:size(u.x[1], 2)
        fids = md.mesh_type.cut_face_nodes[e]

        p, v1, v2 = u.x[1][:,e], u.x[2][:,e], u.x[3][:,e]
        
        du_p  = (Dx_skew[e] * v1 + Dy_skew[e] * v2) # = dv1/dx + dv2/dy
        du_v1 = (Dx_skew[e] * p) # = dp/dx
        du_v2 = (Dy_skew[e] * p) # = dp/dy
        
        # 182 allocations here
        du.x[1][:, e] .+= du_p  + 0.5 * LIFT[e] * flux.x[1][fids] - forcing(md.x.cut[:,e], md.y.cut[:,e], t)
        du.x[2][:, e] .+= du_v1 + 0.5 * LIFT[e] * flux.x[2][fids]
        du.x[3][:, e] .+= du_v2 + 0.5 * LIFT[e] * flux.x[3][fids]
    end

    du .= -du
end


function allocate_rhs_aux_memory(u, uf, rd)

    flux = ArrayPartition(similar(uf.x[1].cartesian), similar(uf.x[2].cartesian), similar(uf.x[3].cartesian))
    dudr = ArrayPartition(similar(u.x[1].cartesian), similar(u.x[2].cartesian), similar(u.x[3].cartesian)) 
    duds = similar(dudr) # ArrayPartition(similar(u.x[1].cartesian), similar(u.x[2].cartesian), similar(u.x[3].cartesian))
    
    Dr = rd.M \ (0.5 * (rd.M * rd.Dr - (rd.M * rd.Dr)'))
    Ds = rd.M \ (0.5 * (rd.M * rd.Ds - (rd.M * rd.Ds)'))

    cart_mem = (; flux, dudr, duds, Dr, Ds)

    cut_flux = ArrayPartition(similar(uf.x[1].cut), similar(uf.x[2].cut), similar(uf.x[3].cut))
    cut_mem = (; flux=cut_flux)

    uBC = zeros(eltype(u), length(u.x))
    return (; cut_mem, cart_mem, uBC)
end