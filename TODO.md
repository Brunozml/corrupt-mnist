# MLOps TODO

## Syllabus

### ✅ S1 setting up a development Environment

- [x] **M1 Command Line\*.** Use basic CL stuff (e.g mkdir, etc.)
- [x] **M2 Package Manager\*.** Uv usage.
- [x] **M3 Editor\*.** VSCode for the win..
- [x] **M4 Deep Learning\*.** Pytorch basics (see notebooks); iterative building.

### S2 Organization and Version Control

- [x] **M5 Git.\*** In Progress; learning best practices as I go.
  - [ ] ~read up docu 30 min a day or smthng
- [x] **M6 Code Structure\*.**
  - [x] Set Up MLOps cookiecutter template.
  - [ ] ~ Read up, understand `Agents.md` and modify.
  - [ ] ~ Create own Cookiecutter template for other stuff
- [ ] **M7 Good coding practice.\*\***
  - [ ] `ruff` for styling
  - [ ] `Typing` data type hints.
- [x] **M8 DVC.**
  - [x] Setup data version control on Google Cloud.
  - [x] DVC for `data/`
  - [ ] ? DVC for `model/`
  - [x] DVC integrated Github Actions for `test_data.py`
  - [ ] DVC integrated Github actions for `test_model.py`
- [ ] **M9 CLIs.**
  - [ ] `typer` CLI for python scripts
  - [ ] `invoke` CLI for Non-python code
- [ ] _EXTRA: `visualization.py`_

### S3 Reproducibility :star: :star: :star:

- [ ] **M10 Docker.\***
  - [X] Set up docker image for `train.py`
  - [ ] Set up docker image for inference
- [x] **M11 Config files**
  - [~] Config file for model
  - [x] Config file for training

### S4 Debugging, Profiling, Logging and Boilerplate

- [~] **M12 Debugging.**
  - [ ] Use VSCode debugger more proactively.
- [ ] ~ **M13 Profiling.** Fire charts.
- [ ] **M13 logging.** :star: :star: :star:
  - [ ] `logger`
  - [ ] `loguru`
  - [ ] W&B, with visuals
- **M15 Boilerplate** (`lighting`; probably skip for now).

### S5 Continuous Integration

- [x] **M16 Unit testing** (with `pytest`)
  - [x] `test_data.py`
  - [~] `test_model.py`
    - [x] `test_model_can_overfit_single_batch()` super nice!
  - [ ] `test_train.py`.
  - [ ] Build `x.py`, then `test_x.py` as I go .. [Example MLOps](https://github.com/SkafteNicki/example_mlops/tree/main) for inspo.
- [x] **M17 Github Actions.**
  - [x] setup basic pipeline with tests for 3 operating systems, 2 python versions.
  - [~] Style checks. Requires M7.
  - [ ] ~ Exercise 12 automatic docker image build. (or do it in the Cloud instead?)
  - [ ] ~ Dependabot setup
  - [ ] ~ Review difference between **workflow, Runner, Job**, and **Action**
- [ ] **M18 Pre-commit.**
  - [ ] ~ Github Actions integration.
- [ ] **M19 Continuous ML.** Requires M13 Logging.

### 🔄 S6 Cloud Computing

- [x] **M20 Cloud setup.**
  - [ ] ~ Record most useful commands in some sort of notes for this repo.
- [~] **M21 Cloud Services.**
  - [x] Setup Bucket with DVC
  - [X] Setup `Artifact Registry` and `Build` services. Requires M10.
  - [ ] ~ Integrate `Build` with Github Actions
  - [ ] Setup `Compute Engine`, i.e. VM, and run raw code.
  - [ ] ~ Vertex AI for training.
  - [x] `Object Viewer` Service account for `data/` tests on Github Actions.
  - [ ] ~ Secrets management for `wandb`

### S7 Model Deployment

- [ ] **M22 Requests and APIs** with `fastapi`
- [ ] **M23 Cloud Deployment.** Serverless functions and serverless containers.
- [ ] ~ **M24 API Testing & Integration testing**
- [ ] ~ **M25 ML Deployment** using`ONNX`
- [ ] ~ **M26 Frontend** using `streamlit`

### S8 Monitoring

- [ ] **M27 Data Drifting**
- [ ] ~ **M28 System Monitoring**

### S9 Scalable Applications

- [ ] **M29 Distributed Data Loading**
- [ ] **M30 Distributed Training** using `lighting.ai`
- [ ] **M31 Scalable Inference**

### S10 Extra learning modules

- [ ] **M32 Documentation**
  - [~] MkDocs
  - [ ] Github Pages
- **M33 Hyperparemeter Optimization.**
- [ ] **M34 HPC**

### Personal Extras

- [~] VSCode Shortcuts
  - [~] `IPython` for interactive building!
  - [ ] `.gif` Animations for `readme.md`
  - [ ] Microsoft Azure setup
  - [ ] CODEX and other CLIs for AIs
  - [] Research THE keyboard setup for seamless transition between Mac and Windows.
  - [] Patience to improve reading documentation.
  - [] ~ `buildx` or similar for building in specific architectures on google cloud?
