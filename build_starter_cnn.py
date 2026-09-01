"""Build a small synthetic-prior spectrogram CNN starter model.

This starter exists so hybrid CNN inference works immediately, but it is not a
substitute for training on real, labeled captures from the user's own devices.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import signal
from cnn_rf import build_cnn, iq_to_spectrogram, CNN_CLASS_NAMES, DEFAULT_SPEC_SIZE

p=argparse.ArgumentParser()
p.add_argument("--out", default="models/house_rf_cnn.pt")
p.add_argument("--epochs", type=int, default=7)
p.add_argument("--samples-per-class", type=int, default=130)
p.add_argument("--seed", type=int, default=570)
a=p.parse_args()

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split
except ImportError as exc:
    raise SystemExit("PyTorch required. Install with: pip install torch") from exc

rng=np.random.default_rng(a.seed); torch.manual_seed(a.seed)
classes=CNN_CLASS_NAMES
sr=4e6; n=16384

def awgn(x,snr_db):
    p=np.mean(np.abs(x)**2)+1e-12
    npow=p/(10**(snr_db/10))
    w=(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))*np.sqrt(npow/2)
    return (x+w).astype(np.complex64)

def qpsk_wave(symbol_rate=5e5, burst=False):
    step=max(2,int(sr/symbol_rate))
    nsym=len(np.arange(0,n,step))
    s=(2*rng.integers(0,2,nsym)-1)+1j*(2*rng.integers(0,2,nsym)-1)
    up=np.zeros(n,dtype=np.complex64); up[::step]=s
    taps=signal.firwin(81,min(0.95,1.3*symbol_rate/(sr/2)))
    x=signal.lfilter(taps,[1],up)
    if burst:
        gate=np.zeros(n); pos=0
        while pos<n:
            on=int(rng.integers(300,1800)); off=int(rng.integers(200,2500)); gate[pos:min(n,pos+on)]=1; pos+=on+off
        x*=gate
    return x

def synth(label):
    t=np.arange(n)/sr
    if label=="broadcast_fm":
        audio=signal.lfilter(signal.firwin(129,0.025),[1],rng.normal(size=n))
        audio/=np.std(audio)+1e-9
        phase=np.cumsum(2*np.pi*(70e3+rng.uniform(-20e3,20e3))*audio/sr)
        x=np.exp(1j*phase)
        bw=200e3
    elif label=="wifi_ofdm":
        nfft=128; cp=16; chunks=[]
        while sum(len(c) for c in chunks)<n:
            bins=np.zeros(nfft,dtype=complex); occ=52
            inds=np.r_[np.arange(1,occ//2+1), np.arange(nfft-occ//2,nfft)]
            bins[inds]=(2*rng.integers(0,2,len(inds))-1)+1j*(2*rng.integers(0,2,len(inds))-1)
            td=np.fft.ifft(bins); chunks.append(np.r_[td[-cp:],td])
        x=np.concatenate(chunks)[:n]
        # packet-like gating
        if rng.random()<0.7:
            gate=np.ones(n); cut=int(rng.uniform(.55,.95)*n); gate[cut:]=0; x*=gate
        bw=1.8e6
    elif label=="bluetooth_fhss":
        bits=2*rng.integers(0,2,n)-1
        shaped=signal.lfilter(signal.windows.gaussian(41,7)/np.sum(signal.windows.gaussian(41,7)),[1],bits)
        phase=np.cumsum(2*np.pi*220e3*shaped/sr)
        x=np.exp(1j*phase)
        gate=(signal.square(2*np.pi*rng.uniform(900,3500)*t,duty=rng.uniform(.15,.55))>0).astype(float)
        x*=gate; bw=900e3
    elif label=="zigbee_like":
        x=qpsk_wave(1.0e6,burst=True); bw=1.8e6
    elif label=="narrowband":
        fdev=rng.uniform(8e3,55e3); mod=signal.lfilter(signal.firwin(81,0.03),[1],rng.normal(size=n))
        phase=np.cumsum(2*np.pi*fdev*mod/(np.std(mod)+1e-9)/sr)
        x=np.exp(1j*phase); bw=80e3
    elif label=="bursty_ism":
        bit_rate=rng.uniform(2e3,60e3); step=max(2,int(sr/bit_rate)); bits=rng.integers(0,2,n//step+1)
        env=np.repeat(bits,step)[:n].astype(float)
        if rng.random()<0.5:
            x=env*np.ones(n,dtype=complex)
        else:
            sym=2*env-1; phase=np.cumsum(2*np.pi*40e3*sym/sr); x=np.exp(1j*phase)*env
        bw=180e3
    elif label=="chirp_css":
        x=np.zeros(n,dtype=complex); seg=1024
        for k in range(0,n,seg):
            nn=min(seg,n-k); tt=np.arange(nn)/sr; bw0=rng.uniform(250e3,1.1e6)
            x[k:k+nn]=np.exp(1j*2*np.pi*((-bw0/2)*tt+0.5*(bw0/(nn/sr))*tt**2))
        bw=900e3
    elif label=="wideband_digital":
        x=qpsk_wave(1.4e6,burst=rng.random()<0.4); bw=2.2e6
    else:
        x=(rng.normal(size=n)+1j*rng.normal(size=n))/np.sqrt(2); bw=2.5e6
    # Random time shift, carrier offset and SNR.
    offset=rng.uniform(-0.12*sr,0.12*sr)
    x=x*np.exp(1j*2*np.pi*offset*t)
    if label!="unknown_noise": x=awgn(x,rng.uniform(7,28))
    return x.astype(np.complex64), offset, bw

X=[]; Y=[]
for yi,label in enumerate(classes):
    for _ in range(a.samples_per_class):
        iq,off,bw=synth(label)
        X.append(iq_to_spectrogram(iq,sr,off,bw,DEFAULT_SPEC_SIZE)); Y.append(yi)
X=torch.from_numpy(np.stack(X)[:,None,:,:]); Y=torch.tensor(Y,dtype=torch.long)
ds=TensorDataset(X,Y); vn=max(1,int(.15*len(ds))); tn=len(ds)-vn
tr,va=random_split(ds,[tn,vn],generator=torch.Generator().manual_seed(a.seed))
tr_loader=DataLoader(tr,batch_size=48,shuffle=True); va_loader=DataLoader(va,batch_size=64)
model=build_cnn(len(classes)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); crit=torch.nn.CrossEntropyLoss()
best=-1; state=None
for epoch in range(1,a.epochs+1):
    model.train(); loss_sum=0; total=0
    for xb,yb in tr_loader:
        opt.zero_grad(set_to_none=True); logits=model(xb); loss=crit(logits,yb); loss.backward(); opt.step()
        loss_sum+=float(loss.detach().item())*len(xb); total+=len(xb)
    model.eval(); ok=0; vt=0
    with torch.no_grad():
        for xb,yb in va_loader:
            pr=model(xb).argmax(1); ok+=int((pr==yb).sum()); vt+=len(xb)
    acc=ok/max(1,vt); print(f"epoch {epoch}/{a.epochs} loss={loss_sum/total:.4f} val_acc={acc:.3f}")
    if acc>best: best=acc; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
torch.save({
    "state_dict":state,
    "class_names":classes,
    "image_size":DEFAULT_SPEC_SIZE,
    "source":"synthetic-prior-spectrogram-cnn",
    "validation_accuracy":float(best),
    "warning":"Synthetic-prior CNN: useful as a starter only. Train on real labeled household captures for meaningful accuracy.",
},a.out)
print("Saved",a.out,"synthetic validation accuracy",best)
