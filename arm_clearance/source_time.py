"""Immutable source time ownership and shared NumPy-only basis primitives."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SourcePartition:
    frame_count: int
    cuts: tuple = ()
    evidence: tuple = ()

    def __post_init__(self):
        if self.frame_count < 1 or tuple(sorted(set(self.cuts))) != self.cuts:
            raise ValueError('Invalid source partition')
        if any(not isinstance(i, (int, np.integer)) or not 0 < i < self.frame_count for i in self.cuts):
            raise ValueError('A source cut must begin an interior frame row')

    @property
    def blocks(self):
        edges=(0,)+self.cuts+(self.frame_count,)
        return tuple(zip(edges[:-1],edges[1:]))

    def first_mask(self):
        mask=np.ones(max(self.frame_count-1,0),dtype=bool)
        if self.cuts:mask[np.asarray(self.cuts)-1]=False
        return mask

    def second_mask(self):
        first=self.first_mask()
        return first[:-1]&first[1:]

    def repair_support(self, frames, center, radius):
        """An interior event owns one segment; a cut bridge owns its two rows."""
        frames=np.asarray(frames,dtype=float)
        if frames.shape!=(self.frame_count,) or not np.isfinite(frames).all() or np.any(np.diff(frames)<=0):
            raise ValueError('Source partition and scene frame rows differ')
        support=np.abs(frames-float(center))<=float(radius)
        if not self.cuts:return support
        at=int(np.searchsorted(frames,center,side='left'))
        exact=at<len(frames) and abs(float(frames[at])-float(center))<1e-7
        if not exact and 0<at<len(frames) and at in self.cuts:
            owned=np.zeros(len(frames),dtype=bool);owned[at-1:at+1]=True
            return support&owned
        row=at if exact else max(0,min(at-1,len(frames)-1))
        segment=int(np.searchsorted(self.cuts,row,side='right'))
        a,b=self.blocks[segment]
        support[:a]=False;support[b:]=False
        return support

    def report(self, frames):
        frames=np.asarray(frames)
        if frames.shape!=(self.frame_count,):raise ValueError('Source report frame count mismatch')
        return {'segments':[[float(frames[a]),float(frames[b-1])] for a,b in self.blocks],
            'cut_rows':list(self.cuts),'source_edges':[[float(frames[i-1]),float(frames[i])] for i in self.cuts],
            'evidence':[{'row':int(i),'relative_wrist_step':float(d),'local_rotation_step_degrees':float(r)}
                        for i,d,r in self.evidence],
            'detection':'isolated shoulder-relative wrist jump >0.8 mean upper-arm lengths and local six-bone rotation step >25 degrees; adjacent-step contrast >4',
            'bridge_repair_scope':'only the two adjacent frame rows',
            'surface_samples_omitted':False}


def detect_source_partition(quaternions, positions):
    q=np.asarray(quaternions,dtype=np.float64);p=np.asarray(positions,dtype=np.float64)
    if q.ndim!=3 or q.shape[1:]!=(6,4) or p.shape!=q.shape[:2]+(3,) or not len(q):
        raise ValueError('Source partition requires six ordered limb poses')
    if not np.isfinite(q).all() or not np.isfinite(p).all():raise ValueError('Nonfinite source pose')
    norms=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(norms<1e-12):raise ValueError('Source quaternion is zero')
    if len(q)==1:return SourcePartition(1)
    length=float(np.median(.5*(np.linalg.norm(p[:,1]-p[:,0],axis=-1)+np.linalg.norm(p[:,4]-p[:,3],axis=-1))))
    if length<=1e-12:raise ValueError('Source upper-arm length is zero')
    relative=p[:,(2,5)]-p[:,(0,3)]
    step=np.max(np.linalg.norm(np.diff(relative,axis=0),axis=-1),axis=-1)/length
    q=q/norms
    angle=np.max(np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[1:]*q[:-1],axis=-1)),0.,1.))),axis=-1)
    neighbor=np.maximum(np.r_[0.,step[:-1]],np.r_[step[1:],0.])
    indices=np.flatnonzero((step>.8)&(angle>25.)&(step>4.*np.maximum(neighbor,1e-12)))
    cuts=tuple(int(i+1) for i in indices)
    evidence=tuple((int(i+1),float(step[i]),float(angle[i])) for i in indices)
    return SourcePartition(len(q),cuts,evidence)


def cubic_basis(frames, spacing):
    t=np.arange(frames)/spacing;k=np.floor(t).astype(int);u=t-k
    weights=np.stack(((1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3),axis=1)/6
    basis=np.zeros((frames,int(k.max())+4))
    basis[np.arange(frames)[:,None],k[:,None]+np.arange(4)]=weights
    return basis


def join_bases(blocks):
    rows=sum(b.shape[0] for b in blocks);columns=sum(b.shape[1] for b in blocks)
    basis=np.zeros((rows,columns));r=c=0
    for block in blocks:
        n,k=block.shape;basis[r:r+n,c:c+k]=block;r+=n;c+=k
    return basis
