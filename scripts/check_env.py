"""Check that the NVIDIA driver, PyTorch's CUDA build, PyG, XGBoost and Java line up.
Run: python scripts/check_env.py"""
import shutil
import subprocess
import sys

OK, BAD = "[ OK ]", "[FAIL]"


def line(status, msg):
    print(f"{status} {msg}")


print(f"Python {sys.version.split()[0]} at {sys.executable}")
print("Virtual env active" if sys.prefix != sys.base_prefix else "WARNING: no virtual env active")
print()

# 1. Driver
driver_cuda = None
if shutil.which("nvidia-smi"):
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    for tok in out.split("|"):
        if "CUDA Version" in tok:
            driver_cuda = tok.split("CUDA Version:")[1].strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    line(OK, f"Driver: {gpu} | supports CUDA up to {driver_cuda}")
else:
    line(BAD, "nvidia-smi not found: NVIDIA driver missing")

# 2. PyTorch
try:
    import torch
    built = torch.version.cuda
    line(OK, f"PyTorch {torch.__version__} built for CUDA {built}")
    if built is None:
        line(BAD, "This is a CPU-only PyTorch build. Reinstall from pytorch.org with a CUDA option.")
    elif driver_cuda and tuple(map(int, built.split("."))) > tuple(map(int, driver_cuda.split("."))):
        line(BAD, f"PyTorch needs CUDA {built} but the driver supports only {driver_cuda}. "
                  "Install a PyTorch build for an older CUDA version.")
    if torch.cuda.is_available():
        x = torch.randn(2048, 2048, device="cuda")
        y = (x @ x).sum().item()
        line(OK, f"GPU compute works on {torch.cuda.get_device_name(0)} "
                 f"(cuDNN {torch.backends.cudnn.version()})")
    else:
        line(BAD, "torch.cuda.is_available() is False: training would run on CPU")
except ImportError:
    line(BAD, "PyTorch not installed")

# 3. PyTorch Geometric
try:
    import torch
    import torch_geometric
    from torch_geometric.nn import SAGEConv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conv = SAGEConv(8, 8).to(dev)
    out = conv(torch.randn(4, 8, device=dev), torch.tensor([[0, 1, 2], [1, 2, 3]], device=dev))
    line(OK, f"PyTorch Geometric {torch_geometric.__version__}: SAGEConv runs on {dev}")
except Exception as e:
    line(BAD, f"PyTorch Geometric problem: {e}")

# 4. XGBoost on GPU
try:
    import numpy as np
    import xgboost as xgb
    X, y = np.random.rand(500, 5), np.random.randint(0, 2, 500)
    xgb.XGBClassifier(n_estimators=5, device="cuda", tree_method="hist").fit(X, y)
    line(OK, f"XGBoost {xgb.__version__} trains on GPU")
except Exception as e:
    line(BAD, f"XGBoost GPU training failed ({e}); run_models.sh will need --gpu removed")

# 5. Java for PySpark
java = shutil.which("java")
if java:
    v = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    line(OK, f"Java: {v}")
else:
    line(BAD, "Java not found (needed by PySpark): sudo apt install openjdk-17-jdk")
