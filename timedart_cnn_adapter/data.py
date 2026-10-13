"""Scan-level folds; exact 144-record identity and split isolation checks."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

ROLES=('train','validation','test')

def canonical(n):
    n=n.replace('\\','/')
    hits=[n.find(v) for v in ('##정상/','##결함/') if v in n]
    if len(hits)!=1:raise ValueError('Unknown class folder: '+n)
    return n[hits[0]:]

def manifest(data_path,split_path):
    doc=json.loads(Path(split_path).read_text(encoding='utf-8'))
    allrows=[];seen_hash=set()
    if set(doc['splits'])!=set(ROLES):raise ValueError('Missing/extra split roles')
    for role in ROLES:
        rows=doc['splits'][role]
        if not rows or any(sum(r['label']==v for r in rows)==0 for v in (0,1)):raise ValueError('Each role requires both classes')
        for r in rows:
            if r['label']!=int('##결함/' in r['file']):raise ValueError('Label mismatch')
            if r['waveform_sha256'] in seen_hash:raise ValueError('Duplicate signal across/within splits')
            seen_hash.add(r['waveform_sha256'])
        allrows+=rows
    reference=json.loads((Path(__file__).parent/'dataset_manifest.json').read_text(encoding='utf-8'))['records']
    if sorted(allrows,key=lambda r:r['file'])!=reference:raise ValueError('Fold must contain exactly the known 144 unique records with original labels/hashes')
    with zipfile.ZipFile(data_path) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.dat') and not n.startswith('__MACOSX/')]
    lookup={canonical(n):n for n in names}
    expected={r['file'] for r in allrows}
    if len(names)!=144 or len(lookup)!=144 or set(lookup)!=expected:raise ValueError('This package requires the exact 144-file data(5)(1).zip / equivalent data(5).zip, not the 274-file dataset.')
    return doc,lookup

def load_roles(data_path,split_path,roles,factor=20):
    if not isinstance(factor,int) or factor<1:raise ValueError('factor must be positive integer')
    doc,lookup=manifest(data_path,split_path);out={};reference=None
    with zipfile.ZipFile(data_path) as z:
        for role in roles:
            if role not in ROLES:raise ValueError(role)
            xs=[];ys=[];names=[]
            for r in doc['splits'][role]:
                a=np.loadtxt(io.BytesIO(z.read(lookup[r['file']])))
                if a.shape==(50001,2):a=a.T
                if a.shape!=(2,50001) or not np.isfinite(a).all():raise ValueError('Invalid shape/nonfinite signal: '+r['file'])
                t,v=a
                if not np.all(np.diff(t)>0):raise ValueError('Invalid time axis')
                if reference is None:reference=t
                if not np.allclose(t,reference,rtol=0,atol=1e-15):raise ValueError('Time axes differ')
                v=v.astype(np.float32)
                if hashlib.sha256(v.tobytes()).hexdigest()!=r['waveform_sha256']:raise ValueError('Signal hash does not match fixed dataset: '+r['file'])
                # Match the attached visualization notebook: float32 -> float64 -> resample_poly.
                q=resample_poly(v.astype(np.float64),1,factor) if factor!=1 else v
                xs.append(q.astype(np.float32));ys.append(r['label']);names.append(r['file'])
            out[role]=(np.stack(xs),np.array(ys,np.int64),names)
    return out
