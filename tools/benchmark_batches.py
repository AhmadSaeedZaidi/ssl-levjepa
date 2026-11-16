"""Fresh disposable optimizer runs on simulator RGB, with measured GPU telemetry."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def telemetry(stop, rows, path):
    with path.open("w") as stream:
        while not stop.is_set():
            result=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True)
            if result.returncode==0:
                v=[float(x.strip()) for x in result.stdout.strip().split(",")]
                row=dict(zip(["gpu_util_percent","temperature_c","power_w","device_used_mib","device_total_mib"],v))
                row["unix_time"]=time.time(); rows.append(row)
                stream.write(json.dumps(row)+"\n"); stream.flush()
            stop.wait(.5)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",default="configs/ssl_sumatra_pilot.yaml")
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for batch in [8,16,32,64]:
        directory=args.output/f"batch_{batch}"
        gpu=[]; stop=threading.Event()
        thread=threading.Thread(target=telemetry,args=(stop,gpu,args.output/f"gpu_{batch}.jsonl"),daemon=True)
        thread.start()
        command=[sys.executable,"-u","tools/train_ssl.py","--config",args.config,"--run-dir",str(directory),
                 "--batch-size",str(batch),"--steps","120","--warmup","20","--benchmark"]
        with (args.output/f"batch_{batch}.log").open("w") as log:
            code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        stop.set(); thread.join(timeout=10)
        record={"batch_size":batch,"exit_code":code,"command":command,"warmup_steps":20,"measured_steps":100}
        if code==0:
            summary=json.loads((directory/"summary.json").read_text())
            rows=[json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()]
            assert len(rows)==120 and summary["nonfinite_count"]==0
            window=[v for v in gpu if rows[19]["unix_time"]<=v["unix_time"]<=rows[-1]["unix_time"]]
            assert window, "No GPU telemetry in timed interval"
            intervals=[rows[i]["wall_elapsed_sec"]-rows[i-1]["wall_elapsed_sec"] for i in range(20,120)]
            record.update({"status":"passed","samples_per_sec":summary["decoded_clips_per_sec"],
                "optimizer_steps_per_sec":summary["steps_per_sec"],"mean_step_sec":statistics.mean(intervals),
                "median_step_sec":statistics.median(intervals),"mean_compute_sec":summary["mean_compute_sec"],
                "mean_data_wait_sec":summary["mean_data_wait_sec"],"peak_allocated_bytes":summary["peak_allocated_bytes"],
                "peak_reserved_bytes":summary["peak_reserved_bytes"],
                "gpu_util_mean_percent":statistics.mean(v["gpu_util_percent"] for v in window),
                "gpu_temperature_max_c":max(v["temperature_c"] for v in window),
                "device_peak_used_mib":max(v["device_used_mib"] for v in window),"telemetry_samples":len(window)})
        else:
            log=(args.output/f"batch_{batch}.log").read_text()
            record["status"]=("out_of_memory" if "OutOfMemoryError" in log or "out of memory" in log else
                              "memory_safety_limit" if "exceeds safety threshold" in log else "error")
            incomplete=directory/"summary.json"
            if incomplete.exists():
                state=json.loads(incomplete.read_text())
                record["completed_steps"]=state["completed_steps"]
                record["measured_steps"]=state["measured_steps"]
        results.append(record)
        (args.output/"results.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(record),flush=True)
    eligible=[r for r in results if r["status"]=="passed" and r["peak_reserved_bytes"]<11*1024**3 and r["device_peak_used_mib"]<11264]
    if not eligible:
        raise RuntimeError("No batch passed the measured memory gate")
    best=max(eligible,key=lambda r:r["samples_per_sec"])
    (args.output/"recommendation.json").write_text(json.dumps({"batch_size":best["batch_size"],"criterion":"highest measured samples/sec with <=11 GiB allocator and device memory gates"},indent=2))


if __name__=="__main__":
    main()
