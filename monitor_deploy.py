import argparse
import subprocess
import json
import time
import sys
from datetime import datetime

def get_active_gcp_project():
    try:
        cmd = ["gcloud", "config", "get-value", "project"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Operation Status Monitor. Polling is done in a clean loop using 'gcloud ai operations describe'."
    )
    parser.add_argument(
        "operation_id",
        type=str,
        help="The Vertex AI Operation ID to monitor (e.g. 1744319835338178560)"
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="GCP Project ID. If not specified, tries to auto-detect using 'gcloud config'."
    )
    parser.add_argument(
        "--region", "-r",
        type=str,
        default="us-central1",
        help="Vertex AI region (default: us-central1)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=15,
        help="Polling interval in seconds (default: 15)"
    )

    args = parser.parse_args()
    
    project_id = args.project
    if not project_id:
        project_id = get_active_gcp_project()
        if not project_id:
            print("❌ Error: GCP Project ID is required. Please specify using --project or configure 'gcloud config set project'.", file=sys.stderr)
            sys.exit(1)
            
    operation_id = args.operation_id
    region = args.region
    interval = args.interval

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Starting monitor for Operation ID: {operation_id}")
    print(f"   Project: {project_id}")
    print(f"   Region:  {region}")
    print(f"   Polling Interval: {interval}s\n")

    last_stage = None

    try:
        while True:
            cmd = [
                "gcloud", "ai", "operations", "describe",
                operation_id,
                f"--project={project_id}",
                f"--region={region}",
                "--format=json"
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Error running gcloud command: {result.stderr.strip()}", file=sys.stderr)
                time.sleep(interval)
                continue
                
            try:
                stdout_clean = result.stdout.strip()
                json_start = stdout_clean.find("{")
                if json_start != -1:
                    stdout_clean = stdout_clean[json_start:]
                data = json.loads(stdout_clean)
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Error parsing JSON: {str(e)}\nRaw output: {result.stdout}", file=sys.stderr)
                time.sleep(interval)
                continue
                
            metadata = data.get("metadata", {})
            stage = metadata.get("deploymentStage")
            done = data.get("done", False)
            error = data.get("error")
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if stage != last_stage:
                print(f"[{current_time}] 🔄 Deployment stage: {stage}")
                last_stage = stage
                
            if error:
                print(f"[{current_time}] ❌ Operation failed with error: {json.dumps(error, indent=2)}")
                break
                
            if done:
                print(f"[{current_time}] 🎉 Operation completed successfully!")
                break
                
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🛑 Monitoring stopped by user.")

if __name__ == "__main__":
    main()
