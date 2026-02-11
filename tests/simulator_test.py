import pytest
import taichi as ti
from simulator.simulator import CNCSimulator

# ========================================================
# Test for simple component initializations
# ========================================================

def test_simulator_initialization():
    try:
        ti.init(arch=ti.cpu)
        sim = CNCSimulator(resolution=128)
    except Exception as e:
        pytest.fail(f"Simulator initialization failed: {e}")


def test_simulator_stock_initialization():
    try:
        ti.init(arch=ti.cpu)
        sim = CNCSimulator(resolution=128)
        sim.initialize_stock_primitive()
    except Exception as e:
        pytest.fail(f"Simulator stock initialization failed: {e}")


def test_simulator_target_initialization():
    try:
        ti.init(arch=ti.cpu)
        sim = CNCSimulator(resolution=128)
        sim.initialize_target_primitive()
    except Exception as e:
        pytest.fail(f"Simulator stock initialization failed: {e}")


def test_simulator_tool_initialization():
    try:
        ti.init(arch=ti.cpu)
        sim = CNCSimulator(resolution=128)
        sim.initialize_tool_primitive()
    except Exception as e:
        pytest.fail(f"Simulator stock initialization failed: {e}")

# ========================================================
# Test for simple move operations
# ========================================================

def test_simulator_move_x_single():
    ti.init(arch=ti.cpu)
    sim = CNCSimulator(resolution=128)
    sim.initialize_stock_primitive()
    sim.initialize_target_primitive()
    sim.initialize_tool_primitive()

    prev = sim.tool_pos.to_numpy().copy()
    sim.move_tool_one_unit(ti.Vector([1.0, 0.0, 0.0]))
    next = sim.tool_pos.to_numpy().copy()

    assert prev[0] + sim.dx == next[0]
    assert prev[1] == next[1]
    assert prev[2] == next[2]


def test_simulator_move_y_single():
    ti.init(arch=ti.cpu)
    sim = CNCSimulator(resolution=128)
    sim.initialize_stock_primitive()
    sim.initialize_target_primitive()
    sim.initialize_tool_primitive()

    prev = sim.tool_pos.to_numpy().copy()
    sim.move_tool_one_unit(ti.Vector([0.0, 1.0, 0.0]))
    next = sim.tool_pos.to_numpy().copy()

    assert prev[0] == next[0]
    assert prev[1] + sim.dx == next[1]
    assert prev[2] == next[2]


def test_simulator_move_z_single():
    ti.init(arch=ti.cpu)
    sim = CNCSimulator(resolution=128)
    sim.initialize_stock_primitive()
    sim.initialize_target_primitive()
    sim.initialize_tool_primitive()

    prev = sim.tool_pos.to_numpy().copy()
    sim.move_tool_one_unit(ti.Vector([0.0, 0.0, 1.0]))
    next = sim.tool_pos.to_numpy().copy()

    assert prev[0] == next[0]
    assert prev[1] == next[1]
    assert prev[2] + sim.dx == next[2]


def test_simulator_move_diagonal_single():
    ti.init(arch=ti.cpu)
    sim = CNCSimulator(resolution=128)
    sim.initialize_stock_primitive()
    sim.initialize_target_primitive()
    sim.initialize_tool_primitive()

    prev = sim.tool_pos.to_numpy().copy()
    sim.move_tool_one_unit(ti.Vector([1.0, 1.0, 0.0]))
    next = sim.tool_pos.to_numpy().copy()

    assert prev[0] + sim.dx == next[0]
    assert prev[1] + sim.dx == next[1]
    assert prev[2] == next[2]


def test_simulator_move_x_multiple():
    ti.init(arch=ti.cpu)
    sim = CNCSimulator(resolution=128)
    sim.initialize_stock_primitive()
    sim.initialize_target_primitive()
    sim.initialize_tool_primitive()

    prev = sim.tool_pos.to_numpy().copy()
    sim.move_tool_one_unit(ti.Vector([1.0, 0.0, 0.0]))
    sim.move_tool_one_unit(ti.Vector([1.0, 0.0, 0.0]))
    sim.move_tool_one_unit(ti.Vector([1.0, 0.0, 0.0]))
    next = sim.tool_pos.to_numpy().copy()

    assert prev[0] + 3 * sim.dx == next[0]
    assert prev[1] == next[1]
    assert prev[2] == next[2]


def test_simulator_move_roundabout():
    ti.init(arch=ti.cpu)
    sim = CNCSimulator(resolution=128)
    sim.initialize_stock_primitive()
    sim.initialize_target_primitive()
    sim.initialize_tool_primitive()

    prev = sim.tool_pos.to_numpy().copy()
    sim.move_tool_one_unit(ti.Vector([1.0, 0.0, 0.0]))
    sim.move_tool_one_unit(ti.Vector([0.0, 1.0, 0.0]))
    sim.move_tool_one_unit(ti.Vector([-1.0, 0.0, 0.0]))
    sim.move_tool_one_unit(ti.Vector([0.0, -1.0, 0.0]))
    next = sim.tool_pos.to_numpy().copy()

    assert prev[0] == next[0]
    assert prev[1] == next[1]
    assert prev[2] == next[2]


# def test_simulator_move_diagonal_invalid_input():

