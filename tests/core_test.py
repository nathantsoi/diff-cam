"""CPU regression checks of the differentiable kernel, independent of RL."""
import numpy as np
import taichi as ti
from simulator.csg_simulator import CSGSimulatorDelta


def make_sim(steps=2):
    ti.init(arch=ti.cpu, offline_cache=False)
    sim = CSGSimulatorDelta(resolution=10, max_steps=steps, stock_size_in=(1,1,1),
                            target_shape='sphere', tool_start=(0.2,0.25,0.5),
                            enforce_speed_limits=False, init_taichi=False, k_init=40)
    sim.tool_radius[None] = 3.175
    sim.tool_height[None] = 25
    sim.set_target_params(radius_mm=9)
    sim.bake_target_grid()
    return sim


def test_delta_gradient_matches_finite_difference():
    sim = make_sim()
    d = np.array([[0.1,0.015,0], [0.12,0.01,0]], dtype=np.float32)
    sim.tool_delta.from_numpy(d)
    with ti.ad.Tape(loss=sim.loss):
        sim.forward(3)
    analytic = float(sim.tool_delta.grad.to_numpy()[0,0])
    values=[]
    eps=1e-3
    for sign in (1,-1):
        perturbed=d.copy(); perturbed[0,0]+=sign*eps
        sim.tool_delta.from_numpy(perturbed)
        sim.loss[None]=0
        sim.forward(3)
        values.append(float(sim.loss[None]))
    numerical=(values[0]-values[1])/(2*eps)
    assert abs(analytic)>1e-5
    np.testing.assert_allclose(analytic,numerical,rtol=0.05,atol=2e-4)


def test_horizontal_hard_carve_is_monotone_and_subdivision_invariant():
    sim=make_sim()
    sim.tool_delta.from_numpy(np.array([[0.4,0,0],[0,0,0]],dtype=np.float32))
    sim.forward_hard(3,clip_speeds=False)
    full=sim.stock.to_numpy()
    assert np.all(full[1:] >= full[:-1]-1e-6)
    sim.tool_delta.from_numpy(np.array([[0.2,0,0],[0.2,0,0]],dtype=np.float32))
    sim.forward_hard(3,clip_speeds=False)
    split=sim.stock.to_numpy()[-1]
    np.testing.assert_array_equal(full[-1]<0,split<0)
