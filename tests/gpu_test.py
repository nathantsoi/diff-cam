import pytest
import taichi as ti

def test_taichi_cpu_support():
    try:
        ti.init(arch=ti.cpu)
        print("Taichi CPU support available")
    except Exception as e:
        pytest.fail(f"Taichi CPU support not available: {e}")

def test_taichi_gpu_support():
    try:
        ti.init(arch=ti.gpu)
        print("Taichi GPU support available")
    except Exception as e:
        pytest.fail(f"Taichi GPU support not available: {e}")




