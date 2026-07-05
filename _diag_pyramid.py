import os, json, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES","2")
from cam.sim_exec import _HardCarveSimulator
from cam.units import inch_to_mm

rd = "runs/CamEnvDiff-v0__train_csg__1__1783216903972"
a = json.load(open(f"{rd}/args.json"))
traj = np.load(f"{rd}/trajectory.npy")
T = len(traj)
sim = _HardCarveSimulator(resolution=32, max_steps=T-1, target_shape=a["target_shape"],
    tool_start=tuple(float(v) for v in traj[0]),
    stock_size_in=tuple(a.get("stock_size_in",(1,1,1))),
    voxel_size_mm=float(a.get("voxel_size_mm",0.5)),
    work_volume_in=tuple(a.get("workspace_in",(16,12,10))))
sim.tool_radius[None]=float(a["tool_radius_mm"]); sim.tool_height[None]=float(a["tool_height_mm"])
sim.holder_radius[None]=inch_to_mm(2.5/2.0); sim.holder_height[None]=float(sim.work_volume_mm[2])
sim.set_target_params(radius_mm=float(a["target_radius_mm"]),height_mm=float(a["target_height_mm"]),
    half_size_mm=float(a["target_radius_mm"]),center=(0.5,0.5,0.5))
sim.bake_target_grid(); sim.set_target_volume(); sim.enforce_z_floor[None]=1
deltas=np.diff(traj,axis=0); padded=np.zeros((sim.max_steps,3),dtype=np.float32)
padded[:len(deltas)]=deltas; sim.tool_delta.from_numpy(padded); sim.forward_hard(T)

stock = sim.stock.to_numpy()[-1]
print("stock shape",stock.shape)
target = sim.target.to_numpy()       # <0 = target
remaining = stock < 0
tgt = target < 0
Nx,Ny,Nz = remaining.shape
# per-z target retention
print("remaining shape",remaining.shape, "tgt shape",tgt.shape,"target vox",int(tgt.sum()),"remaining",int(remaining.sum()))
print("z(layer) : target_vox  remaining_target  removed_target  waste_remaining")
for z in range(0,Nz,4):
    tv = int(tgt[:,:,z].sum())
    rt = int((tgt[:,:,z]&remaining[:,:,z]).sum())
    rm = tv-rt
    wr = int((~tgt[:,:,z]&remaining[:,:,z]).sum())
    print(f"{z:3d}     : {tv:6d}      {rt:6d}          {rm:6d}        {wr:6d}")
# total
print("TOTAL target removed (gouge) =", int((tgt & ~remaining).sum()))
print("TOTAL waste remaining (under) =", int((~tgt & remaining).sum()))
print("dice(remaining,target) =", 2*int((tgt&remaining).sum())/(int(remaining.sum())+int(tgt.sum())))
