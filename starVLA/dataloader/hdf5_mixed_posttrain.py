from __future__ import annotations
import os
from pathlib import Path
try:
    import h5py
except ImportError:
    h5py = None
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MixedHDF5PostTrainDataset(Dataset):
    def __init__(self, data_cfg, mode="train", **kwargs):
        self.cache_root = Path(str(data_cfg.get("cache_root", ""))) if data_cfg.get("cache_root") else None
        self.horizon=int(data_cfg.get("action_horizon",15)); self.image_size=tuple(int(x) for x in data_cfg.get("obs_image_size",[224,224]))
        self.default_instruction=str(data_cfg.get("instruction","Perform the demonstrated manipulation."))
        if h5py is None and not (self.cache_root and self.cache_root.exists()): raise ImportError("h5py is required for direct HDF5 training")
        self.roots=[Path(str(x)) for x in data_cfg.get("roots", [])]; self.files=[]
        if self.cache_root and (self.cache_root / "index.npz").exists():
            self.cache_items=np.load(self.cache_root/'index.npz')['items']; self.items=list(range(len(self.cache_items))); self.files=[]; self._handles={}; print(f'[MixedHDF5PostTrainDataset] cache samples={len(self.items)} image_size={self.image_size}',flush=True); return
        for r in self.roots: self.files += sorted(r.rglob("*.hdf5"))
        if not self.files: raise FileNotFoundError(self.roots)
        self.default_instruction=str(data_cfg.get("instruction","Perform the demonstrated manipulation.")); self._handles={}; self._pid=os.getpid(); self.items=[]
        old_limit = int(data_cfg.get("old_demo_limit", 50)); new_limit = int(data_cfg.get("new_demo_limit", 200))
        root_seen = {str(r): 0 for r in self.roots}; self.bad_files=[]
        self.action_q01=np.asarray(data_cfg.get("action_q01",[-0.310128,-0.595406,-0.350657,-2.81489,-0.515495,1.659108,-1.510504,0.0]),np.float32)
        self.action_q99=np.asarray(data_cfg.get("action_q99",[0.410902,0.987285,0.275238,-1.025782,0.35958,2.867717,1.707022,1.0]),np.float32)
        self.state_q01=np.asarray(data_cfg.get("state_q01",self.action_q01),np.float32); self.state_q99=np.asarray(data_cfg.get("state_q99",self.action_q99),np.float32)
        self.normalize=bool(data_cfg.get("normalize_actions_states",True))
        for fi,p in enumerate(self.files):
            is_old = "banana-in-bowl" in str(p).lower(); root_key=next((str(r) for r in self.roots if str(p).startswith(str(r))),str(self.roots[0]))
            try: fctx=h5py.File(p,"r")
            except (OSError, IOError): self.bad_files.append(str(p)); continue
            with fctx as f:
                for d in sorted(f["data"]):
                    limit=old_limit if is_old else new_limit
                    if root_seen[root_key] >= limit: continue
                    n=int(f["data"][d]["actions"].shape[0]); self.items += [(fi,d,t) for t in range(max(0,n-self.horizon+1))]
                    root_seen[root_key] += 1
        lim=data_cfg.get("max_samples"); self.items=self.items if lim in (None,"") else self.items[:int(lim)]
        print(f"[MixedHDF5PostTrainDataset] files={len(self.files)} trajectories={root_seen} samples={len(self.items)} bad_files={len(self.bad_files)} image_size={self.image_size}",flush=True)
    def __len__(self): return len(self.items)
    def _f(self,fi):
        if os.getpid()!=self._pid: self._handles={}; self._pid=os.getpid()
        k=str(self.files[fi])
        if k not in self._handles: self._handles[k]=h5py.File(k,"r",rdcc_nbytes=64*1024*1024,rdcc_nslots=200003,swmr=True)
        return self._handles[k]
    def __getitem__(self,i):
        if self.cache_root and self.files == []:
            k=int(self.cache_items[self.items[i]]); p=self.cache_root/f'sample_{k:07d}'; z=np.load(str(p)+'.npz'); ims=[Image.open(str(p)+f'_{j}.jpg').convert('RGB') for j in (0,1)]; return {'image':ims,'lang':self.default_instruction,'state':z['state'],'action':z['action']}
        fi,d,t=self.items[i]; g=self._f(fi)["data"][d]; io=g["obs"]["image_obs"]
        imgs=[Image.fromarray(np.asarray(io[k][t],np.uint8),"RGB").resize(self.image_size,Image.Resampling.BILINEAR) for k in ("over_shoulder_left_camera","wrist_cam")]
        a=np.asarray(g["actions"][t:t+self.horizon],np.float32); o=g["obs"]["proprio_obs"]; s=np.r_[o["arm_joint_pos"][t],o["gripper_pos"][t]].astype(np.float32)[None]
        if self.normalize:
            a=np.clip(2*(a-self.action_q01)/np.maximum(self.action_q99-self.action_q01,1e-6)-1,-1,1)
            s=np.clip(2*(s-self.state_q01)/np.maximum(self.state_q99-self.state_q01,1e-6)-1,-1,1)
        return {"image":imgs,"lang":self.default_instruction,"state":s,"action":a}
def collate_fn(batch): return batch
def build_dataset(data_cfg,mode="train",**kwargs): return MixedHDF5PostTrainDataset(data_cfg,mode,**kwargs)
