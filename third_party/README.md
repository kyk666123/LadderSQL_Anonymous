# third_party

Vendored external dependencies. Do not edit unless you intend to change
LadderSQL's behaviour; see `../NOTICE` for provenance and licensing.

## agentlightning

A snapshot of [Agent Lightning](https://github.com/microsoft/agent-lightning)
(MIT, Copyright (c) Microsoft Corporation) reporting version `0.3.1`. That
version was never published to PyPI, and the copy carries local modifications
needed by the multi-turn RL loop, so it is vendored rather than pinned as a
dependency.

Make it importable before training:

```bash
export PYTHONPATH="$(git rev-parse --show-toplevel)/third_party:$PYTHONPATH"
python -c "import agentlightning; print(agentlightning.__version__)"   # 0.3.1
```
