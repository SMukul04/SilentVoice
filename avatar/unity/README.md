# Unity Avatar Runtime

### Purpose
Unity is the 3D rendering/runtime layer for SilentVoice.

### Current status
Phase 8.1 infrastructure only.

### Responsibilities
Unity:
* renders avatar
* executes animations
* reports animation completion
* eventually communicates with browser/backend

Backend:
* text processing
* sign resolution
* sign sequencing
* semantic orchestration
* animation asset resolution

### Future communication
```text
FastAPI
   ↓
Browser JavaScript
   ↓
Unity WebGL
   ↓
Avatar Runtime
```

Communication is NOT implemented yet.
