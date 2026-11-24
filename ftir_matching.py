"""FTIR reference search for single spectra, independent of CACTNet.
PEPE binary layout: https://openreadout.github.io/openreadout/formats/perkinelmer-sp.html
"""
from __future__ import annotations
import io
import struct
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / 'spectral_library'
MAX_SPECTRUM_BYTES = 5 * 1024 * 1024
FULL_RANGE = "Full-range Matching"
ADAPTIVE_RANGE = "Adaptive-range Matching"
MIN_COVERAGE = 0.70


def blocks(buf):
    pos = 0
    while pos < len(buf):
        if len(buf) - pos < 6:
            raise ValueError('Incomplete SP block header.')
        key, size = struct.unpack_from('<Hi', buf, pos)
        pos += 6
        if size < 0 or pos + size > len(buf):
            raise ValueError('Invalid SP block length.')
        yield key, buf[pos:pos+size]
        pos += size


def read_sp(data):
    if not data.startswith(b'PEPE2D constant interval DataSet file'):
        raise ValueError('Only PerkinElmer PEPE 2D constant-interval SP files are supported.')
    datasets = [v for k, v in blocks(data[44:]) if k == 120]
    if len(datasets) != 1:
        raise ValueError('Expected exactly one spectrum in the SP file.')
    fields = dict(blocks(datasets[0]))
    def typed(key, tag, size=None):
        v = fields.get(key, b'')
        if len(v) < 2 or struct.unpack_from('<H', v)[0] != tag or (size is not None and len(v) != size + 2):
            raise ValueError(f'Invalid or missing SP field {key}.')
        return v[2:]
    def unit(key):
        v = typed(key, 0x7523)
        if len(v)<2 or struct.unpack_from('<H',v)[0] != len(v)-2:
            raise ValueError('Invalid SP unit field.')
        return v[2:].decode('ascii').strip()
    first, last = struct.unpack('<dd', typed(35698, 0x751d, 16))
    step, = struct.unpack('<d', typed(35700, 0x751b, 8))
    count, = struct.unpack('<I', typed(35701, 0x752b, 4))
    v = typed(35708, 0x7516)
    if len(v)<4 or struct.unpack_from('<I',v)[0] != len(v)-4 or len(v)-4 != 8*count or count < 2:
        raise ValueError('SP point count does not match the spectrum data.')
    if not np.isfinite([first,last,step]).all() or not np.isclose(abs(last-first),abs(step)*(count-1),rtol=1e-4,atol=0.01):
        raise ValueError('SP wavenumber interval is inconsistent.')
    if unit(35703).lower() not in {'cm-1','cm^-1','1/cm'}:
        raise ValueError('SP x axis is not wavenumber (cm-1).')
    yunit = unit(35704).upper()
    if yunit not in {'A','%T'}:
        raise ValueError('Unsupported SP signal unit; expected A or %T.')
    return np.linspace(first,last,count), np.frombuffer(v[4:],dtype='<f8').copy(), 'Absorbance' if yunit=='A' else 'Transmittance (%)'


def read_text(data, signal):
    for encoding in ('utf-8-sig','gb18030','utf-16'):
        try:
            text=data.decode(encoding)
            break
        except UnicodeError:
            continue
    else:
        raise ValueError('Unable to decode the text spectrum.')
    frame=pd.read_csv(io.StringIO(text),sep=r'[,;\s]+',engine='python',header=None,comment='#',dtype=str)
    if frame.shape[1]!=2:
        raise ValueError('Use exactly two columns: wavenumber (cm-1), signal.')
    numeric=frame.apply(pd.to_numeric,errors='coerce')
    # Permit one header row only; reject corrupted numeric data.
    if len(numeric) and numeric.iloc[0].isna().all():
        numeric=numeric.iloc[1:]
    if numeric.isna().any().any():
        raise ValueError('Non-numeric values found in the spectrum.')
    return numeric.iloc[:,0].to_numpy(),numeric.iloc[:,1].to_numpy(),signal


def read_spectrum(data, filename, signal='Absorbance'):
    if len(data)>MAX_SPECTRUM_BYTES:
        raise ValueError('Spectrum file exceeds 5 MiB.')
    suffix=Path(filename).suffix.lower()
    if suffix=='.sp':
        x,y,signal=read_sp(data)
    elif suffix in {'.csv','.txt'}:
        x,y,signal=read_text(data,signal)
    else:
        raise ValueError('Supported formats: SP, CSV, TXT.')
    if len(x)<100 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Spectrum has too few points or contains invalid values.')
    if signal in {'Transmittance (%)','Transmittance (fraction)'}:
        if np.any(y<=0):
            raise ValueError('Transmittance must be positive for absorbance conversion.')
        y=-np.log10(y/(100 if signal=='Transmittance (%)' else 1))
    elif signal!='Absorbance':
        raise ValueError('Unsupported signal type.')
    order=np.argsort(x)
    x,y=x[order],y[order]
    if np.any(np.diff(x)<=0):
        raise ValueError('Duplicate wavenumbers are not supported.')
    return x,y,signal


class ReferenceLibrary:
    def __init__(self, root=ROOT):
        with np.load(Path(root)/'reference_absorbance.npz',allow_pickle=False) as z:
            self.axis=z['axis'].astype(np.float64)
            self.raw=z['raw_absorbance'].astype(np.float32,copy=False)
            ids=z['library_id'].astype(str)
        self.meta=pd.read_csv(Path(root)/'library_manifest.csv',keep_default_na=False,dtype={'library_id':str})
        if not np.array_equal(ids,self.meta.library_id.to_numpy()) or self.raw.shape!=(len(self.meta),len(self.axis)):
            raise ValueError('Reference matrix and metadata do not align.')
        if not np.isfinite(self.raw).all() or not np.isfinite(self.axis).all() or np.any(np.diff(self.axis)<=0):
            raise ValueError('Invalid reference spectra.')
        self.centered=self.raw-self.raw.mean(axis=1,keepdims=True)
        norm=np.linalg.norm(self.centered,axis=1,keepdims=True)
        if np.any(norm<=1e-10):
            raise ValueError('Constant reference spectrum found.')
        self.centered/=norm

    def match(self, data, filename, signal='Absorbance', mode=FULL_RANGE):
        x, y, unit = read_spectrum(data, filename, signal)
        if mode not in (FULL_RANGE, ADAPTIVE_RANGE):
            raise ValueError('Unknown matching mode.')
        full = self.axis
        if mode == FULL_RANGE:
            # Preserve the original full-range acceptance and scoring rules.
            if x[0] > full[0] or x[-1] < full[-1] or np.max(np.diff(x)) > 10:
                raise ValueError('Spectrum must cover 650–4000 cm⁻¹ with no gaps larger than 10 cm⁻¹.')
            lo, hi = 0, len(full)
        else:
            # Use only library grid points inside the measured range; no extrapolation.
            lo = int(np.searchsorted(full, x[0], side='left'))
            hi = int(np.searchsorted(full, x[-1], side='right'))
            if hi-lo < 2:
                raise ValueError('No sufficient overlapping wavenumber range.')
            coverage = (full[hi-1]-full[lo]) / (full[-1]-full[0])
            if coverage < MIN_COVERAGE:
                raise ValueError(f'Overlapping range covers only {coverage:.1%} of the reference range; at least 70% is required.')
            # Reject gaps that intersect the actual matching range (including its boundaries).
            gaps = np.diff(x)[(x[:-1] < full[hi-1]) & (x[1:] > full[lo])]
            if gaps.size and np.max(gaps) > 10:
                raise ValueError('The matching range contains a gap larger than 10 cm⁻¹.')
        axis = full[lo:hi]
        coverage = float((axis[-1]-axis[0]) / (full[-1]-full[0]))
        query = np.interp(axis, x, y)
        q = query - query.mean()
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n <= 1e-10:
            raise ValueError('Spectrum is constant or invalid.')
        q = (q/n).astype(np.float32)
        if lo == 0 and hi == len(full):
            scores = np.clip(self.centered @ q, -1, 1)
        else:
            # Recenter and normalize each reference on the SAME overlap.
            # Chunk processing avoids a second full library-sized allocation.
            scores = np.full(len(self.meta), -np.inf, dtype=np.float32)
            for start in range(0, len(self.meta), 256):
                block = self.raw[start:start+256, lo:hi].copy()
                block -= block.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(block, axis=1)
                valid = np.isfinite(norms) & (norms > 1e-10)
                if valid.any():
                    block[valid] /= norms[valid, None]
                    target = scores[start:start+len(block)]
                    target[valid] = np.clip(block[valid] @ q, -1, 1)
        order = np.argsort(-scores, kind='stable')
        order = order[np.isfinite(scores[order])]
        if not len(order):
            raise ValueError('No usable reference spectra in the matching range.')
        hits = self.meta.iloc[order][['material_group','material_subtype','source','sample_id','library_id']].copy()
        hits['similarity'] = scores[order]
        hits = hits.drop_duplicates('material_group').head(5).reset_index(drop=True)
        hits.insert(0, 'rank', np.arange(1, len(hits)+1))
        best = int(order[0])
        return {'hits': hits, 'axis': axis.copy(), 'query': query,
                'reference': self.raw[best, lo:hi].copy(), 'unit': unit,
                'count': len(order), 'mode': mode, 'range_min_cm1': float(axis[0]),
                'range_max_cm1': float(axis[-1]), 'coverage_percent': coverage*100,
                'fingerprint_complete': bool(axis[0] <= 650 and axis[-1] >= 1500)}
