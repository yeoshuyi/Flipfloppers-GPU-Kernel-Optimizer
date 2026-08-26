import subprocess, json, time, pathlib
RUNS = pathlib.Path("/scratch/techjam2/runs")

def submit(candidate, shape, mode="bench"):
    out = subprocess.run(
        ["sbatch", "--parsable", "/scratch/work/jobs/bench.sbatch",
         candidate, shape, mode],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()

def poll(job_id, timeout_s=1500):
    result = RUNS / f"{job_id}.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if result.exists():
            return json.loads(result.read_text())
        state = subprocess.run(["sacct", "-j", job_id, "-n", "-o", "State"],
                               capture_output=True, text=True).stdout
        if any(s in state for s in ("FAILED", "TIMEOUT", "CANCELLED")):
            return {"error": state.strip(),
                    "log": (RUNS / f"{job_id}.out").read_text()[-4000:]}
        time.sleep(5)
    subprocess.run(["scancel", job_id])
    return {"error": "timeout"}
