import sqlite3
import hashlib, base64
import argparse
import shutil
import tempfile
import zipfile
import urllib.request
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as TPE
from pathlib import Path
from urllib.parse import quote
from collections import defaultdict
from itertools import cycle
from multiprocessing import Process
from Bio import SeqIO
from tqdm import tqdm
import os, sys
import queue
import subprocess
from contextlib import contextmanager
import decimal
import fcntl
import functools
import struct
import threading
import time

# --- Logging functions
def message(msg):
    print(f"[*] {msg}", file = sys.stderr)

def ok(msg):
    print(f"[✔] {msg}", file = sys.stderr)

def warning(msg):
    print(f"[!] {msg}", file = sys.stderr)

def error(msg, exception = None, fatal = True):
    print(f"[✘] {msg}", file = sys.stderr)
    if exception:
        print(f"[✘] {exception}", file = sys.stderr)
    if fatal:
        sys.exit(1)

# --- Handle GPUs ---

def get_framework_gpu_count():
    import importlib.util
    def is_installed(package_name):
        return importlib.util.find_spec(package_name) is not None
    if is_installed("torch"):
        import torch
        return torch.cuda.device_count()
    if is_installed("tensorflow"):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus)
    return 0

def default_gpus():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        if cuda_visible_devices.strip() in ("", "-1"):
            return []
        return [x.strip() for x in cuda_visible_devices.split(",")]
    num_gpus = get_framework_gpu_count()
    return [str(i) for i in range(num_gpus)]

# --- GPU locking ---

# An exclusive lock on the first byte of the device node, held while waiting for the GPU lock
TURNSTILE = struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 1, 0)
DEF_MAX_WAIT = 12 * 3600
# Seconds between attempts to take a lock: a blocking lock could not give up at a deadline
LOCK_POLL = 1

class GPUWaitTimeout(Exception):
    pass

@functools.cache
def device_node(gpu):
    """The device node of a GPU, numbered by its minor number, which need not be its index."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "-q"], text = True, stderr = subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"Could not query GPU {gpu} with nvidia-smi: {e}")
    for line in out.splitlines():
        key, _, val = line.partition(":")
        if key.strip() == "Minor Number" and val.strip().isdigit():
            return f"/dev/nvidia{val.strip()}"
    raise RuntimeError(f"Could not find the device node of GPU {gpu}")

def lock_holder(path):
    """
    The PID that took the GPU lock on the device node, if it is visible from here. The lock may
    outlive it, held by a child it started.
    """
    st = os.stat(path)
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    try:
        with open("/proc/locks") as f:
            for line in f:
                # e.g. "2: FLOCK  ADVISORY  WRITE 12345 00:05:1111 0 EOF"; waiters have "->" after the number
                fields = line.split()
                if fields[1] == "FLOCK" and fields[5] == dev and int(fields[4]) > 0:
                    return int(fields[4])
    except (OSError, IndexError, ValueError):
        pass
    return None

def duration(seconds):
    for unit, size in (("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} s"

def locked_by(path):
    holder = lock_holder(path)
    if holder is None:
        return ""
    if os.path.exists(f"/proc/{holder}"):
        return f" (locked by PID {holder})"
    return f" (locked by PID {holder}, which has exited, but a process it started still runs)"

@contextmanager
def gpu_lock(gpu, max_wait = DEF_MAX_WAIT):
    """
    Holds a GPU among the processes that take this lock, waiting while another one holds it, for
    up to max_wait seconds, after which GPUWaitTimeout is raised.

    The lock is a flock on the GPU's device node, which every process that can use the GPU sees,
    and the kernel releases it whenever the process dies. It is compatible with the lock taken by
    run_alphafold3 and can also be taken from the shell with `flock /dev/nvidiaN <command>`.

    Yields the descriptor that holds the lock. Passed to a child with pass_fds, it keeps the GPU
    locked until the child exits too, even if the caller dies first.

    A process that has to wait first queues on a turnstile, an exclusive lock on the first byte of
    the device node. A process that releases the GPU and wants it back has to pass the turnstile,
    which lets the waiting process in first.
    """
    path = device_node(gpu)
    lock_fd = os.open(path, os.O_RDONLY)
    try:
        # A write lock needs a descriptor open for writing
        turnstile_fd = os.open(path, os.O_RDWR)
        try:
            deadline = time.monotonic() + max_wait
            waiting = False
            for take in (lambda: fcntl.fcntl(turnstile_fd, fcntl.F_OFD_SETLK, TURNSTILE),
                         lambda: fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)):
                while True:
                    try:
                        take()
                        break
                    except BlockingIOError:
                        pass
                    if time.monotonic() >= deadline:
                        raise GPUWaitTimeout(f"GPU {gpu} is still busy{locked_by(path)} after {duration(max_wait)} of waiting")
                    if not waiting:
                        tqdm.write(f"[*] GPU {gpu} is busy{locked_by(path)}, waiting", file = sys.stderr)
                        waiting = True
                    time.sleep(LOCK_POLL)
        finally:
            os.close(turnstile_fd)
        yield lock_fd
    finally:
        # Closing rather than unlocking: a child still holding the descriptor keeps the lock
        os.close(lock_fd)

# --- Constants ---
MODELS = [
    'simplefold_100M', 'simplefold_360M', 'simplefold_700M', 
    'simplefold_1.1B', 'simplefold_1.6B', 'simplefold_3B'
]

DEF_GPUS = default_gpus()
DEF_TAU = decimal.Decimal("0.1")
DEF_BATCH = 100
DEF_SEEDS = [123]
DEF_MODEL = 'simplefold_360M'
DEF_STEPS = 500

# --- Helper Functions ---

@contextmanager
def get_log(log_path):
    if not log_path:
        yield subprocess.DEVNULL
        return
    f = open(log_path, 'w', encoding='utf-8')
    try:
        yield f
    finally:
        f.close()

def check_tau(tau):
    if not type(tau) == decimal.Decimal:
        return False
    sign, digits, exponent = tau.as_tuple()
    return sign == 0 and exponent >= -2 and len(digits) - exponent <= 5

def clean_record_seq(record_seq):
    return record_seq.upper().strip().replace("*", "").replace("-", "")

def get_hash_and_quote(record, desc):
    clean_seq = str(clean_record_seq(record.seq))
    raw_hash = hashlib.sha256(clean_seq.encode('ascii')).digest()
    name = record.description if desc else record.id
    return base64.urlsafe_b64encode(raw_hash).decode().rstrip("="), quote(name, safe = "-_")

def download_file(url, output_path):
    """
    Downloads a file, unless it exists. The file is written under a temporary name and renamed
    when complete, so that an interrupted download is not taken for a complete one.
    """
    if output_path.exists():
        return
    output_path.parent.mkdir(parents = True, exist_ok = True)
    message(f"Downloading: {url} -> {output_path}")
    with tempfile.NamedTemporaryFile(dir = output_path.parent, prefix = f".{output_path.name}.", suffix = ".part", delete = False) as out_file:
        part_path = Path(out_file.name)
        try:
            with urllib.request.urlopen(url) as response:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, out_file)
            if expected is not None and out_file.tell() != int(expected):
                raise IOError(f"got {out_file.tell()} bytes instead of {expected}")
        except BaseException:
            part_path.unlink(missing_ok = True)
            raise
    part_path.chmod(0o644)
    os.replace(part_path, output_path)

def download_dir(url, output_path):
    """
    Downloads and unpacks a zip archive of a single directory, unless it exists. The directory is
    unpacked next to the destination and renamed when complete.
    """
    if output_path.exists():
        return
    output_path.parent.mkdir(parents = True, exist_ok = True)
    with tempfile.TemporaryDirectory(dir = output_path.parent, prefix = f".{output_path.name}.") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "local.zip"
        dir_path = tmp_path / "local"
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dir_path)
        src = next(dir_path.iterdir())
        src.chmod(0o755)
        os.rename(src, output_path)

# --- Initialization Logic ---

def data_paths(data_dir):
    data_path = Path(data_dir).resolve()
    ckpt_path = data_path / "ckpt"
    torch_path = data_path / "torch"
    cache_path = data_path / "cache"
    sf_repo_path = data_path / "ml-simplefold-main"
    esm_ckpt_path = torch_path / "hub" / "checkpoints"
    esm_repo_path = torch_path / "hub" / "facebookresearch_esm_main"
    latent_path = ckpt_path / "simplefold_1.6B.ckpt"
    plddt_path = ckpt_path / "plddt.ckpt"
    return data_path, ckpt_path, torch_path, cache_path, sf_repo_path, esm_ckpt_path, esm_repo_path, latent_path, plddt_path

def model_paths(model, data_dir):
    data_path = Path(data_dir).resolve()
    model_path = data_path / "models" / f"{model}.ckpt" if model != "simplefold_1.6B" else None
    db_path = data_path / "databases" / f"{model}.sq3"
    return model_path, db_path

def model_url(model):
    return f"https://ml-site.cdn-apple.com/models/simplefold/{model}.ckpt"

def launch_init(data_dir, models = []):
    """General environment setup, the 1.6B latent model and the models' checkpoints and databases."""
    data_path, ckpt_path, torch_path, cache_path, sf_repo_path, esm_ckpt_path, esm_repo_path, latent_path, plddt_path = data_paths(data_dir)

    # 2. Build General Download Queue
    file_tasks = [
        (model_url("simplefold_1.6B"), latent_path),
        ("https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt", plddt_path),
        ("https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl", cache_path / "ccd.pkl"),
        ("https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt", cache_path / "boltz1_conf.ckpt"),
        ("https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt", esm_ckpt_path / "esm2_t36_3B_UR50D.pt"),
        ("https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt", esm_ckpt_path / "esm2_t36_3B_UR50D-contact-regression.pt")
    ]
    # simplefold_1.6B is the latent model, downloaded above
    for model in models:
        model_path, db_path = model_paths(model, data_dir)
        if model_path:
            file_tasks.append((model_url(model), model_path))
    dir_tasks = [
        ("https://github.com/apple/ml-simplefold/archive/refs/heads/main.zip", sf_repo_path),
        ("https://github.com/facebookresearch/esm/archive/refs/heads/main.zip", esm_repo_path)
    ]
    # 3. Parallel Downloads for General Init
    message("Starting parallel downloads...")
    with TPE(max_workers = len(file_tasks) + len(dir_tasks)) as executor:
        futures = {}
        for url, target in file_tasks:
            futures[executor.submit(download_file, url, target)] = url
        for url, target in dir_tasks:
            futures[executor.submit(download_dir, url, target)] = url
        failed = []
        with tqdm(total = len(file_tasks) + len(dir_tasks), leave = True) as progress:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    failed.append(futures[future])
                    error(f"Failed to download {futures[future]}", e, fatal = False)
                progress.update(1)
    if failed:
        error(f"{len(failed)} download(s) failed. Run simplefold_init again to retry them", fatal = True)

    for model in models:
        init_database(model, data_dir)
    ok("Initialization complete" + (f": {', '.join(models)}" if models else ""))

def open_cache(db_path, read_only = False):
    """
    Connects to a cache database. A read-only connection never writes to it, but waits for the
    commits of others writing to it at the same time.
    """
    if not read_only:
        return sqlite3.connect(db_path)
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri = True, timeout = 30)
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.OperationalError as e:
        error(f"Cannot read {db_path} read-only ({e}). A writer that crashed may have left a journal, which only a writer can roll back: run once without --read-only-cache", fatal = True)
    return conn

def fetch_cif(conn, seq_hash, seed, tau, steps):
    found = conn.execute("SELECT cif FROM simplefold WHERE seq_hash = ? AND seed = ? AND tau = ? AND steps = ?", (seq_hash, seed, str(tau), steps)).fetchone()
    return found[0] if found else None

def init_database(model, data_dir):
    model_path, db_path = model_paths(model, data_dir)
    db_path.parent.mkdir(parents = True, exist_ok = True)
    message(f"Initializing database at {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS simplefold (seq_hash TEXT, seed INT, tau DECIMAL(5,2), steps INT, cif TEXT, PRIMARY KEY (seq_hash, seed, tau, steps))")
        conn.execute("CREATE TABLE IF NOT EXISTS proteins (seq_hash TEXT PRIMARY KEY, seq TEXT UNIQUE NOT NULL)")

# --- Inference Logic ---

def simplefold_exe():
    """SimpleFold's executable from the environment we run in, which need not be activated."""
    return shutil.which("simplefold", path = os.path.dirname(sys.executable)) or "simplefold"

def run_gpu_worker(batch, gpu, output_path, data_dir, model, model_path, log, seed, tau, steps, max_wait):

    data_path, ckpt_path, torch_path, cache_path, sf_repo_path, esm_ckpt_path, esm_repo_path, latent_path, plddt_path = data_paths(data_dir)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        tmp_ckpt_path = tmp_path / "ckpt"
        configs_path = tmp_path / "configs"
        fasta_path = tmp_path / "fasta"
        pred_path = tmp_path / "pred"

        pred_path.mkdir()

        (pred_path / cache_path.name).symlink_to(cache_path, target_is_directory = True)
        configs_path.symlink_to(sf_repo_path / "configs", target_is_directory = True)

        tmp_ckpt_path.mkdir()
        if model_path:
            (tmp_ckpt_path / model_path.name).symlink_to(model_path)
        (tmp_ckpt_path / latent_path.name).symlink_to(latent_path)
        (tmp_ckpt_path / plddt_path.name).symlink_to(plddt_path)

        fasta_path.mkdir()
        for seq_hash, basename, rec in batch:
            SeqIO.write(rec, fasta_path / f"{basename}.fasta", "fasta")
        env = os.environ.copy()
        # Number the GPUs as nvidia-smi does, so that the locked device is the one used
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TORCH_HOME"] = str(torch_path)
        cmd = [
            simplefold_exe(), "--simplefold_model", model, "--num_steps", str(steps),
            "--tau", str(tau), "--nsample_per_protein", "1", "--backend", "torch",
            "--plddt", "--seed", str(seed), "--fasta_path", fasta_path.name,
            "--output_dir", pred_path.name, "--ckpt_dir", ckpt_path.name
        ]
        # The child inherits the lock, so that the GPU stays locked while it runs, even if we die
        with gpu_lock(gpu, max_wait) as lock_fd:
            subprocess.run(cmd, env = env, check = True, cwd = tmp_path, stdout = log, stderr = log, pass_fds = [lock_fd])
        output = []
        for seq_hash, seq_quote, record in batch:
            generated_cif = pred_path / f"predictions_{model}" / f"{seq_quote}_sampled_0.cif"
            output_cif = output_path / f"{seq_quote}_{seed}.cif"
            if generated_cif.exists():
                shutil.move(generated_cif, output_cif)
            else:
                output_cif = None
            output.append((seq_hash, seq_quote, record, output_cif))
        return output

def launch_run(input_fasta, output_dir, data_dir, model, log_file, batch_size, seed, tau, steps, gpus, desc, no_cache, read_only_cache = False, max_wait = DEF_MAX_WAIT):

    if not gpus:
        error("No GPUs allocated", fatal = True)
    try:
        for gpu in gpus:
            device_node(gpu)
    except RuntimeError as e:
        error(str(e), fatal = True)

    if not check_tau(tau):
        error(f"--tau must be a non-negative decimal number with at most two digits after the dot", fatal = True)
    
    output_path = Path(output_dir).resolve() / model / str(seed)
    output_path.mkdir(parents = True, exist_ok = True)

    model_path, db_path = model_paths(model, data_dir)

    if not Path(data_dir).exists():
        error(f"Data directory {data_dir} not found. Run 'simplefold_init' first", fatal = True)
    if not db_path.exists():
        error(f"Database {db_path} not found. Run 'simplefold_init -m' first", fatal = True)
    if model_path and not model_path.exists():
        error(f"Model file {model_path} not found. Run 'simplefold_init -m' first", fatal = True)

    # With --read-only-cache, cached structures are used, but new ones are not added
    write_cache = not no_cache and not read_only_cache
    # Writing also needs the directory, for the journal
    if write_cache and not (os.access(db_path, os.W_OK) and os.access(db_path.parent, os.W_OK)):
        error(f"Cannot write to {db_path}: use --read-only-cache", fatal = True)

    print(f"[*] Checking the fasta file")
    to_analyze = {}
    all_ids = set()
    with open_cache(db_path, read_only_cache) as conn:
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq_hash, seq_quote = get_hash_and_quote(record, desc)
            if seq_quote in all_ids:
                error(f"Duplicated record id {record.id}", fatal = True)
            all_ids.add(seq_quote)
            if no_cache:
                to_analyze[record.id] = seq_hash, seq_quote
                continue
            cif = fetch_cif(conn, seq_hash, seed, tau, steps)
            if cif:
                output_file_path = output_path / f"{seq_quote}_{seed}.cif"
                output_file_path.write_text(cif)
            else:
                to_analyze[record.id] = seq_hash, seq_quote
    print(f"[✔] A total of {len(all_ids)} sequences, {len(to_analyze)} to analyze")

    def process_fasta():
        to_predict = {}
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq_hash, seq_quote = to_analyze.pop(record.id, (None, None))
            if seq_hash:
                record.seq = clean_record_seq(record.seq)
                to_predict[record.id] = seq_hash, seq_quote, record
                if len(to_predict) == batch_size:
                    yield to_predict
                    to_predict = {}
        if to_predict:
            yield to_predict

    # Free GPUs; None once every GPU has been given up on
    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    gpus_left = [len(gpus)]
    gpus_left_lock = threading.Lock()

    def wrapper(batch, log):
        while (gpu := gpu_queue.get()) is not None:
            try:
                result = run_gpu_worker(batch, gpu, output_path, data_dir, model, model_path, log, seed, tau, steps, max_wait)
            except GPUWaitTimeout as e:
                # Give up on this GPU and leave the batch to the others
                error(str(e), fatal = False)
                with gpus_left_lock:
                    gpus_left[0] -= 1
                    if gpus_left[0] == 0:
                        error("Gave up on all GPUs", fatal = False)
                        gpu_queue.put(None)
                continue
            except Exception:
                gpu_queue.put(gpu)
                raise
            gpu_queue.put(gpu)
            return result
        # Pass the sentinel on to the other waiting batches
        gpu_queue.put(None)
        return []

    def check_futures(futures, conn, max_num = 1):
        successes = set()
        assert max_num > 0
        while len(futures) >= max_num:
            futures_done, futures = concurrent.futures.wait(futures, return_when = concurrent.futures.FIRST_COMPLETED)
            for future in futures_done:
                try:
                    output = future.result()
                except Exception as e:
                    error(f"A batch failed. Check log file.", e, fatal = True)
                for seq_hash, seq_quote, record, output_cif in output:
                    if output_cif:
                        successes.add(record.id)
                        if write_cache:
                            cif = output_cif.read_text() 
                            conn.execute("INSERT OR IGNORE INTO proteins VALUES (?, ?)", (seq_hash, str(record.seq)))
                            conn.execute("INSERT OR IGNORE INTO simplefold VALUES (?, ?, ?, ?, ?)", (seq_hash, seed, str(tau), steps, cif))
                if write_cache:
                    conn.commit()
        return futures, successes

    ok = True
    num_gpus = len(gpus)
    with get_log(log_file) as log, TPE(max_workers = num_gpus) as executor, open_cache(db_path, read_only_cache) as conn, tqdm(total = len(to_analyze)) as progress_bar:
        all_names = set(to_analyze.keys())
        success_names = set()
        futures = set()
        for to_predict in process_fasta():
            all_names |= set(to_predict.keys())
            futures, successes = check_futures(futures, conn, max_num = num_gpus)
            success_names |= successes
            futures.add(executor.submit(wrapper, to_predict.values(), log))
            progress_bar.update(len(successes))
        futures, successes = check_futures(futures, conn)
        success_names |= successes
        progress_bar.update(len(successes))
        ok = True
        for name in all_names - success_names:
            error(f"No structure produced for {name}", fatal = False)
            ok = False
        if not ok:
            error("Something went wrong")
    print(f"[✔] All done")

def launch_select(input_dir, output_dir, soft_link, no_seed_suffix = False):
    import gemmi, re

    files = {}
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents = True, exist_ok = True)

    seed_re = re.compile(r'_[0-9]+$')
    trim_seed_suffix = not no_seed_suffix
    col = 'B_iso_or_equiv'
    for cif_path in input_dir_path.rglob("*.cif"):
        doc = gemmi.cif.read_file(str(cif_path))
        atoms = doc[0].find('_atom_site.', [ col ])
        plddts = [ float(a[col]) for a in atoms ]
        score = sum(plddts) / len(plddts)
        name = cif_path.stem
        if trim_seed_suffix:
            name = seed_re.sub('', name)
        if name not in files or files[name][0] < score:
            files[name] = score, cif_path
    for name, (score, source_path) in files.items():
        output_path = output_dir_path / f"{name}.cif"
        if soft_link:
            output_path.symlink_to(source_path.relative_to(output_dir_path))
        else:
            shutil.copy(source_path, output_path)

# --- CLI Entries ---

def init_cli():
    def models_arg(arg):
        if arg.strip() == "all":
            return MODELS
        models = [x.strip() for x in arg.split(',')]
        for model in models:
            if model not in MODELS:
                raise argparse.ArgumentTypeError(f"unknown model {model} (choose from {', '.join(MODELS)} or all)")
        return list(dict.fromkeys(models))
    parser = argparse.ArgumentParser(description = "SimpleFold wrapper: Initialize")
    parser.add_argument("-D", "--data-dir", required = True, help = "Base directory for data")
    parser.add_argument("-m", "--models", type = models_arg, default = [], help = f"Models to set up, comma-separated, or all ({', '.join(MODELS)}) (optional)")
    args = parser.parse_args()
    launch_init(args.data_dir, args.models)

def run_cli():
    def list_of_int_arg(arg):
        return [ int(x) for x in arg.split(',') ]
    def list_of_str_arg(arg):
        return [x.strip() for x in arg.split(',')]
    def tau_arg(arg):
        tau = decimal.Decimal(arg)
        if check_tau(tau):
            return tau
        else:
            raise argparse.ArgumentTypeError(f"--tau must be a non-negative decimal number with at most two digits after the dot")

    parser = argparse.ArgumentParser(description = "SimpleFold wrapper: Run")
    parser.add_argument("-i", "--input", required = True, help = "Path to input sequences")
    parser.add_argument("-O", "--output", required = True, help = "Output directory")
    parser.add_argument("-D", "--data-dir", required = True, help = "Base directory for data")
    parser.add_argument("-m", "--model", required = True, choices = MODELS, default = DEF_MODEL, help = f"Model name to use for inference [{DEF_MODEL}]")
    parser.add_argument("-g", "--gpus", type=list_of_str_arg, default = DEF_GPUS, help=f"GPUs to use [{','.join(DEF_GPUS)}]")
    parser.add_argument("-s", "--seeds", type = list_of_int_arg, default = DEF_SEEDS, help = f"Seeds [{','.join(map(str, DEF_SEEDS))}]")
    parser.add_argument("-b", "--batch", type = int, default = DEF_BATCH, help = f"Batch size [{DEF_BATCH}]")
    parser.add_argument("-l", "--log", type = str, help = "Raw log file")
    parser.add_argument("--tau", type = tau_arg, default = DEF_TAU, help = f"Tau [{DEF_TAU}]")
    parser.add_argument("--steps", type = int, default = DEF_STEPS, help = f"Number of steps [{DEF_STEPS}]")
    parser.add_argument("--desc", action = 'store_true', help = "Consider the whole fasta header is the record name")
    parser.add_argument("--no-cache", action = 'store_true', help = "Do not use cache")
    parser.add_argument("--read-only-cache", action = 'store_true', help = "Use cached structures, but do not add new ones to the cache")
    parser.add_argument("--max-wait", type = float, default = DEF_MAX_WAIT, help = f"Seconds to wait for a busy GPU before giving up on it [{DEF_MAX_WAIT}, i.e. {duration(DEF_MAX_WAIT)}]")
    args = parser.parse_args()
    if args.max_wait < 0:
        parser.error("--max-wait must not be negative")
    for seed in args.seeds:
        print(f"Starting with seed {seed}...")
        launch_run(
            args.input, args.output, args.data_dir, args.model, args.log,
            batch_size = args.batch, seed = seed, tau = args.tau, steps = args.steps, desc = args.desc, gpus = args.gpus, no_cache = args.no_cache,
            read_only_cache = args.read_only_cache, max_wait = args.max_wait
        )

def select_cli():
    parser = argparse.ArgumentParser(description = "SimpleFold wrapper: Select")
    parser.add_argument("-I", "--input", required = True, help = "Directory containing 'run' outputs")
    parser.add_argument("-O", "--output", required = True, help = "Output directory for the best models")
    parser.add_argument("-l", "--soft-link", action = 'store_true', help = "Soft link instead of hard copy")
    parser.add_argument("--no-seed-suffix", action = 'store_true', help = "The input files do not contain the seed as the suffix")
    args = parser.parse_args()
    launch_select(args.input, args.output, args.soft_link, no_seed_suffix = args.no_seed_suffix)
