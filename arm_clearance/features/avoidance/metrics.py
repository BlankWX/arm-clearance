"""Existing proxy metrics, separate from real surface acceptance."""
import numpy as np
from ...numerics import project_parameters

def full_metrics(model,solver,delta):
    delta=project_parameters(model,delta)
    points,orientation,elbows,wrists=model.state(delta)
    _,_,_,contact=solver.contacts(points,orientation,False)
    angle=np.linalg.norm(delta[:,:6],axis=-1)*180/np.pi
    def stats(value):return {'mean':float(value.mean()),'p95':float(np.percentile(value,95)),'max':float(value.max())}
    body=contact['body'];worst=np.maximum(body.max((1,2)),np.maximum(contact['mutual_frames'],contact['own_frames']))
    return {'frames':model.n,'body_max_mm':float(body.max()*1000),'mutual_max_mm':contact['mutual_max']*1000,
        'own_hand_arm_max_mm':contact['own_max']*1000,'frames_over_1mm':int(np.sum(worst>.001)),
        'frames_over_3mm':int(np.sum(worst>.003)),
        'upper_arm_degrees':stats(angle[:,(0,3)]),'forearm_degrees':stats(angle[:,(1,4)]),'wrist_degrees':stats(angle[:,(2,5)]),
        'wrist_translation_mm':stats(np.linalg.norm(delta[:,6:]*model.length,axis=-1)*1000),
        'elbow_displacement_mm':stats(np.linalg.norm(elbows-model.elbow_orig,axis=-1)*1000),
        'rotation_acceleration_deg_per_frame2':stats(np.linalg.norm(np.diff(delta[:,:6],n=2,axis=0),axis=-1)*180/np.pi),
        'worst_frame_indices':np.argsort(worst)[-20:][::-1].tolist(),
        'source_fidelity':solver.source_deviation_metrics(points,orientation,elbows,wrists),
        'source_motion':solver.source_motion_metrics(orientation,elbows,wrists)}
