# Requantize the F16 2D weights of a vision projector GGUF to Q8_0 (rows must be multiples of 32); other tensors and all
# metadata are copied unchanged. Usage: python3 mmproj_q8.py mmproj-F16.gguf mmproj-Q8_0.gguf
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q, GGUFValueType as V

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
w = GGUFWriter(dst, arch=r.fields["general.architecture"].contents())
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == V.ARRAY else None)
nq = nk = 0
for t in r.tensors:
    if t.tensor_type == Q.F16 and len(t.shape) == 2 and int(t.shape[0]) % 32 == 0:
        w.add_tensor(t.name, gguf.quants.quantize(np.asarray(t.data, dtype=np.float32), Q.Q8_0), raw_dtype=Q.Q8_0)
        nq += 1
    else:
        w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)
        nk += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"{nq} tensors -> Q8_0, {nk} kept; {os.path.getsize(src) / 1e6:.0f} MB -> {os.path.getsize(dst) / 1e6:.0f} MB")
