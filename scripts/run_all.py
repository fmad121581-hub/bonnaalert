"""Run the live part of the pipeline: forecast -> score -> alerts.

    python scripts/run_all.py            # real forecast
    python scripts/run_all.py --demo     # synthetic storm (no internet needed)
    python scripts/run_all.py --llm      # also reword orange/red alerts with an LLM
    python scripts/run_all.py --replay data/events/2023-09-21_rain_raw.json   # replay a past storm
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script, *args):
    print(f"\n=== {script} {' '.join(args)}")
    subprocess.run([sys.executable, str(HERE / script), *args], check=True, cwd=HERE)


if __name__ == "__main__":
    extra = ["--demo"] if "--demo" in sys.argv else []
    if "--replay" in sys.argv:
        extra = ["--replay", str(Path(sys.argv[sys.argv.index("--replay") + 1]).resolve())]
        if "--scale-to" in sys.argv:
            extra += ["--scale-to", sys.argv[sys.argv.index("--scale-to") + 1]]
    run("fetch_forecast.py", *extra)
    run("score.py")
    run("alerts.py", *(["--llm"] if "--llm" in sys.argv else []))
