import taichi as ti

##############################################################################
# Initialization kernels for stock, target, and tool primitives.
# These are called by the environment after reset to set up the initial SDFs and tool parameters.
##############################################################################


@ti.func
def sphere_sdf(p, center, radius):
    return (p - center).norm() - radius


@ti.func
def box_sdf(p, center, half_size):
    d = ti.abs(p - center) - half_size
    return ti.max(d.x, ti.max(d.y, d.z))


@ti.func
def cylinder_sdf(p, bottom_x, bottom_y, bottom_z, radius, height):
    """Cylinder with bottom face at (bottom_x, bottom_y, bottom_z), extending upward."""
    d_h = ti.Vector([p.x - bottom_x, p.y - bottom_y]).norm() - radius
    d_z_bottom = bottom_z - p.z
    d_z_top = p.z - (bottom_z + height)
    d_z = ti.max(d_z_bottom, d_z_top)
    return ti.max(d_h, d_z)


@ti.func
def pyramid_sdf(p, cx, cy, base_z, half_base, height):
    t = (p.z - base_z) / height
    d_bottom = base_z - p.z
    d_top = p.z - (base_z + height)
    allowed = half_base * (1.0 - t)
    dx = ti.abs(p.x - cx) - allowed
    dy = ti.abs(p.y - cy) - allowed
    d_sides = ti.max(dx, dy)
    return ti.max(d_bottom, ti.max(d_top, d_sides))


@ti.func
def boolean_subtract(stock_dist, tool_dist):
    return ti.max(stock_dist, -tool_dist)


# Math
@ti.func
def smooth_max(a, b, k):
    return (1.0 / k) * ti.log(ti.exp(k * a) + ti.exp(k * b))



