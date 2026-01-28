import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "corrupt_mnist"
PYTHON_VERSION = "3.12"


# ============================================================================
# CODE QUALITY TASKS
# ============================================================================


@task
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff linter on source code.

    Args:
        fix: Automatically fix issues if True (default: False)

    Example:
        invoke lint          # Check only
        invoke lint --fix    # Fix issues
    """
    fix_flag = "--fix" if fix else ""
    print("🔍 Running ruff linter...")
    ctx.run(f"uv run ruff check src/ tests/ {fix_flag}", echo=True, pty=not WINDOWS)


@task
def format_code(ctx: Context, check: bool = False) -> None:
    """Format code with ruff formatter.

    Args:
        check: Only check formatting without changing files (default: False)

    Example:
        invoke format-code         # Format files
        invoke format-code --check # Check only (CI mode)
    """
    check_flag = "--check" if check else ""
    print("✨ Formatting code with ruff...")
    ctx.run(f"uv run ruff format src/ tests/ {check_flag}", echo=True, pty=not WINDOWS)


@task
def type_check(ctx: Context) -> None:
    """Run mypy type checker."""
    print("🔬 Running mypy type checker...")
    ctx.run("uv run mypy src/", echo=True, pty=not WINDOWS)


@task(pre=[lint, type_check])
def check(ctx: Context) -> None:
    """Run all code quality checks (lint + type check).

    Note: Uses 'pre' to automatically run lint and type_check first.
    This is a task dependency pattern!
    """
    print("✅ All code quality checks passed!")


# ============================================================================
# TESTING TASKS
# ============================================================================


@task
def test(ctx: Context, verbose: bool = False, coverage: bool = True) -> None:
    """Run tests with optional coverage.

    Args:
        verbose: Show verbose test output (default: False)
        coverage: Generate coverage report (default: True)

    Example:
        invoke test                      # Normal tests with coverage
        invoke test --verbose            # Verbose output
        invoke test --no-coverage        # Skip coverage report
    """
    verbose_flag = "-v" if verbose else ""

    if coverage:
        print("🧪 Running tests with coverage...")
        ctx.run(f"uv run coverage run -m pytest tests/ {verbose_flag}", echo=True, pty=not WINDOWS)
        ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)
    else:
        print("🧪 Running tests...")
        ctx.run(f"uv run pytest tests/ {verbose_flag}", echo=True, pty=not WINDOWS)


@task
def test_data(ctx: Context) -> None:
    """Run only data tests."""
    print("🧪 Testing data pipeline...")
    ctx.run("uv run pytest tests/test_data.py -v", echo=True, pty=not WINDOWS)


@task
def test_model(ctx: Context) -> None:
    """Run only model tests."""
    print("🧪 Testing model...")
    ctx.run("uv run pytest tests/test_model.py -v", echo=True, pty=not WINDOWS)


# ============================================================================
# DATA & TRAINING TASKS
# ============================================================================


@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    print("📊 Preprocessing data...")
    ctx.run(f"uv run python -m {PROJECT_NAME}.data", echo=True, pty=not WINDOWS)


@task
def train_model(ctx: Context) -> None:
    """Train model."""
    print("🚂 Training model...")
    ctx.run(f"uv run python -m {PROJECT_NAME}.train", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, model_path: str = "models/model.pth") -> None:
    """Evaluate trained model.

    Args:
        model_path: Path to model checkpoint (default: models/model.pth)

    Example:
        invoke evaluate
        invoke evaluate --model-path=models/best_model.pth
    """
    print(f"📈 Evaluating model: {model_path}")
    ctx.run(f"uv run python -m {PROJECT_NAME}.evaluate --model-path {model_path}", echo=True, pty=not WINDOWS)


# ============================================================================
# DVC (DATA VERSION CONTROL) TASKS
# ============================================================================


@task
def dvc_pull(ctx: Context) -> None:
    """Pull data and models from DVC remote storage."""
    print("📥 Pulling data from DVC remote...")
    ctx.run("uv run dvc pull", echo=True, pty=not WINDOWS)


@task
def dvc_push(ctx: Context) -> None:
    """Push data and models to DVC remote storage."""
    print("📤 Pushing data to DVC remote...")
    ctx.run("uv run dvc push", echo=True, pty=not WINDOWS)


@task
def dvc_status(ctx: Context) -> None:
    """Check DVC status."""
    print("📊 Checking DVC status...")
    ctx.run("uv run dvc status", echo=True, pty=not WINDOWS)


# ============================================================================
# DOCKER TASKS
# ============================================================================


@task
def docker_build(ctx: Context, image: str = "all", progress: str = "plain") -> None:
    """Build docker images.

    Args:
        image: Which image to build: 'train', 'api', or 'all' (default: all)
        progress: Docker build progress output style (default: plain)

    Example:
        invoke docker-build                    # Build all images
        invoke docker-build --image=train      # Build only train image
        invoke docker-build --progress=auto    # Use auto progress
    """
    images_to_build = {
        "train": ("train:latest", "dockerfiles/train.dockerfile"),
        "api": ("api:latest", "dockerfiles/api.dockerfile"),
    }

    if image == "all":
        to_build = images_to_build.items()
    elif image in images_to_build:
        to_build = [(image, images_to_build[image])]
    else:
        print(f"❌ Unknown image: {image}. Choose from: train, api, all")
        return

    for name, (tag, dockerfile) in to_build:
        print(f"🐳 Building {name} image...")
        ctx.run(f"docker build -t {tag} . -f {dockerfile} --progress={progress}", echo=True, pty=not WINDOWS)


@task
def docker_run_train(ctx: Context) -> None:
    """Run training in Docker container."""
    print("🐳 Running training in Docker...")
    ctx.run("docker run --rm train:latest", echo=True, pty=not WINDOWS)


@task
def docker_run_api(ctx: Context, port: int = 8000) -> None:
    """Run API server in Docker container.

    Args:
        port: Port to expose (default: 8000)

    Example:
        invoke docker-run-api
        invoke docker-run-api --port=8080
    """
    print(f"🐳 Running API on port {port}...")
    ctx.run(f"docker run --rm -p {port}:{port} api:latest", echo=True, pty=not WINDOWS)


# ============================================================================
# DOCUMENTATION TASKS
# ============================================================================


@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    print("📚 Building documentation...")
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context, port: int = 8000) -> None:
    """Serve documentation locally.

    Args:
        port: Port to serve on (default: 8000)

    Example:
        invoke serve-docs
        invoke serve-docs --port=8080
    """
    print(f"📚 Serving docs on http://localhost:{port}")
    ctx.run(
        f"uv run mkdocs serve --config-file docs/mkdocs.yaml --dev-addr=0.0.0.0:{port}",
        echo=True,
        pty=not WINDOWS,
    )


# ============================================================================
# COMPOSITE TASKS (Full Workflows)
# ============================================================================


@task(pre=[check, test])
def ci(ctx: Context) -> None:  # noqa: ARG001
    """Run full CI pipeline (checks + tests).

    This demonstrates task dependencies using 'pre'.
    First runs: check (which runs lint + type_check)
    Then runs: test
    Finally: prints success message
    """
    print("✅ CI pipeline passed!")


@task(pre=[dvc_pull, train_model])
def train_pipeline(ctx: Context) -> None:
    """Full training pipeline: pull data → train → evaluate.

    This chains multiple tasks together!
    """
    print("📈 Running evaluation...")
    evaluate(ctx)
    print("✅ Training pipeline complete!")


@task(pre=[check, test, docker_build])
def build_all(ctx: Context) -> None:  # noqa: ARG001
    """Run quality checks, tests, and build Docker images.

    Full pre-deployment pipeline!
    """
    print("✅ Full build complete!")


@task
def clean(ctx: Context) -> None:
    """Clean generated files and caches."""
    print("🧹 Cleaning up...")
    patterns = [
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "*.pyc",
        ".coverage",
        "htmlcov",
        "build",
        "dist",
        "*.egg-info",
    ]
    for pattern in patterns:
        ctx.run(f"find . -type d -name '{pattern}' -exec rm -rf {{}} + 2>/dev/null || true", echo=True)
        ctx.run(f"find . -type f -name '{pattern}' -delete 2>/dev/null || true", echo=True)
    print("✅ Cleanup complete!")
