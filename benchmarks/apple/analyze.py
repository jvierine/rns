import hashlib
from pathlib import Path
import h5py
import numpy as np
root = Path(__file__).resolve().parent
reference = np.fromfile(root / 'reference.bin', dtype='<f8').reshape(-1, 7)
with h5py.File(root / 'cpu_benchmark.h5', 'a') as out:
    for name, filename in [('aos', 'output.bin'), ('soa', 'output_soa.bin')]:
        actual = np.fromfile(root / filename, dtype='<f4').reshape(-1, 7)
        assert actual.shape == reference.shape and np.isfinite(actual).all()
        norms = np.linalg.norm(reference, axis=0)
        error = np.linalg.norm(actual-reference, axis=0) / np.maximum(norms, 1e-300)
        assert np.max(error) < 5e-5, error
        timing = (root / ('metal_timing.txt' if name == 'aos' else 'metal_soa_timing.txt')).read_text()
        rows = [line.split() for line in timing.splitlines() if line.startswith('threads ')]
        group = out.require_group(name)
        for key, values in [('threads', [int(x[1]) for x in rows]), ('wall_seconds', [float(x[3]) for x in rows]), ('gpu_seconds', [float(x[5]) for x in rows]), ('relative_l2', error)]:
            if key in group: del group[key]
            group.create_dataset(key, data=values)
        group.attrs['device'] = timing.splitlines()[0]
        print(name, 'wall', group['wall_seconds'][:], 'relative L2', error)
    for source in ['hlle.metal', 'hlle_soa.metal', 'run.swift', '../../src/bin/benchmark_metal.rs']:
        out.attrs['sha256:' + source] = hashlib.sha256((root/source).read_bytes()).hexdigest()
    out.attrs['scope'] = 'HLLE face flux only; excludes reconstruction, source terms, MPI, and time stepping. GPU buffers resident; command submission and waiting included in wall timings.'
    out.attrs['validated'] = 1
