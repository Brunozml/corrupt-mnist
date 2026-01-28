# Invoke CLI Guide

This project uses [Invoke](https://www.pyinvoke.org/) for task automation. Invoke is a Python-based task execution tool, similar to Make but with Python's power and readability.

## Quick Start

```bash
# List all available tasks
invoke --list

# Get help on a specific task
invoke --help docker-build

# Run a task
invoke lint

# Run a task with parameters
invoke lint --fix
invoke docker-build --image=train
invoke test --no-coverage
```

## Why Invoke?

| Tool | Best For |
|------|----------|
| **invoke** | Orchestrating workflows, multi-step builds, MLOps pipelines |
| **typer** | Python CLIs with complex argument parsing |
| **Makefile** | Simple builds, traditional C/C++ projects |

**Invoke advantages:**
- ✅ Cross-platform (handles Windows/Unix differences automatically)
- ✅ Written in Python (no shell scripting!)
- ✅ Task dependencies and chaining
- ✅ Type hints and IDE support
- ✅ Better error handling than shell scripts

## Task Categories

### 🔍 Code Quality
```bash
invoke lint                    # Check code style
invoke lint --fix              # Fix code style issues
invoke format-code             # Format code
invoke format-code --check     # Check formatting (CI mode)
invoke type-check              # Run mypy type checker
invoke check                   # Run ALL checks (lint + type-check)
```

### 🧪 Testing
```bash
invoke test                    # Run all tests with coverage
invoke test --verbose          # Verbose test output
invoke test --no-coverage      # Skip coverage report
invoke test-data               # Run only data tests
invoke test-model              # Run only model tests
```

### 📊 Data & Training
```bash
invoke dvc-pull                # Pull data from DVC remote
invoke dvc-push                # Push data to DVC remote
invoke dvc-status              # Check DVC status
invoke preprocess-data         # Preprocess raw data
invoke train-model             # Train the model
invoke evaluate                # Evaluate trained model
invoke evaluate --model-path=models/best.pth  # Evaluate specific model
```

### 🐳 Docker
```bash
invoke docker-build                    # Build all Docker images
invoke docker-build --image=train      # Build only train image
invoke docker-build --image=api        # Build only API image
invoke docker-build --progress=auto    # Use auto progress display
invoke docker-run-train                # Run training in container
invoke docker-run-api                  # Run API server
invoke docker-run-api --port=8080      # Run API on custom port
```

### 📚 Documentation
```bash
invoke build-docs              # Build documentation
invoke serve-docs              # Serve docs locally on port 8000
invoke serve-docs --port=8080  # Serve docs on custom port
```

### 🚀 Composite Workflows
```bash
invoke ci                      # Run CI pipeline (check + test)
invoke train-pipeline          # Full training: pull data → train → evaluate
invoke build-all               # Quality checks + tests + Docker builds
invoke clean                   # Clean generated files and caches
```

## Key Invoke Patterns

### 1. Task with Parameters
```python
@task
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff linter."""
    fix_flag = "--fix" if fix else ""
    ctx.run(f"uv run ruff check src/ tests/ {fix_flag}", echo=True, pty=not WINDOWS)
```

**Usage:**
- `invoke lint` - Check only
- `invoke lint --fix` - Fix issues
- `invoke lint --no-fix` - Explicitly disable fixing

### 2. Task Dependencies (pre)
```python
@task(pre=[lint, type_check])
def check(ctx: Context) -> None:
    """Run all checks."""
    print("✅ All checks passed!")
```

**What happens when you run `invoke check`:**
1. Runs `lint` task
2. Runs `type_check` task
3. Runs `check` task body
4. If any pre-task fails, execution stops

### 3. Multiple Tasks in Sequence
```bash
# Run multiple tasks in order
invoke lint test docker-build

# Equivalent to:
# 1. invoke lint
# 2. invoke test  (only if lint passes)
# 3. invoke docker-build (only if test passes)
```

### 4. Conditional Logic
```python
@task
def docker_build(ctx: Context, image: str = "all") -> None:
    if image == "all":
        # Build all images
    elif image in ["train", "api"]:
        # Build specific image
    else:
        print(f"Unknown image: {image}")
        return
```

## Common Workflows

### Before Committing Code
```bash
invoke check test
```
Runs linting, type checking, and all tests.

### Training a New Model
```bash
invoke train-pipeline
```
Pulls data, trains model, and evaluates it.

### Preparing for Deployment
```bash
invoke build-all
```
Runs quality checks, tests, and builds Docker images.

### Daily Development
```bash
# Format and fix code
invoke format-code lint --fix

# Run fast checks
invoke check

# Run specific tests
invoke test-model --verbose
```

## Tips & Best Practices

1. **Always check task list first:**
   ```bash
   invoke --list
   ```

2. **Get help on any task:**
   ```bash
   invoke --help task-name
   ```

3. **Use `--echo` in ctx.run() to see commands:**
   ```python
   ctx.run("command", echo=True)  # Shows what's being executed
   ```

4. **Boolean flags:**
   - `--flag` sets to True
   - `--no-flag` sets to False
   - Default value defined in function signature

5. **String/Int parameters:**
   ```bash
   invoke task --param=value
   invoke task --param value     # Also works
   ```

6. **Chain tasks for efficiency:**
   ```bash
   invoke check test docker-build  # All in one command
   ```

## Adding New Tasks

1. Open `tasks.py`
2. Add your task:
   ```python
   @task
   def my_task(ctx: Context, param: str = "default") -> None:
       """Task description (shown in --help)."""
       print(f"Running with {param}")
       ctx.run("your-command-here", echo=True, pty=not WINDOWS)
   ```
3. Test it:
   ```bash
   invoke my-task --param=test
   ```

## Debugging

**Task not running?**
- Check `invoke --list` to see if it's registered
- Verify the `@task` decorator is present
- Check for Python syntax errors

**Command failing?**
- Use `echo=True` to see the exact command
- Run the command directly in terminal to debug
- Check if `uv` environment is activated

**Import errors?**
- Run `uv sync` to install dependencies
- Check virtual environment is active

## Learn More

- [Invoke Documentation](https://docs.pyinvoke.org/)
- [Task Patterns](https://docs.pyinvoke.org/en/stable/concepts/invoking-tasks.html)
- [Configuration](https://docs.pyinvoke.org/en/stable/concepts/configuration.html)
