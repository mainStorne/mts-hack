"""Обход ошибки gliner2==2.1.0: _attach_entity_attributes индексирует CPU-тензор логитов индексом на CUDA."""
import inspect
import textwrap

import gliner2.models.boundary.engine as _engine

_BAD = "torch.tensor(rows, dtype=torch.long, device=device)"


def apply():
    cls = _engine.BoundaryExtractor
    if getattr(cls, "_cpu_index_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(cls._attach_entity_attributes))
    assert _BAD in src, "gliner2 changed: patch no longer applies"
    ns = {}
    exec(src.replace(_BAD, "torch.tensor(rows, dtype=torch.long)"), vars(_engine), ns)
    cls._attach_entity_attributes = ns["_attach_entity_attributes"]
    cls._cpu_index_patched = True
