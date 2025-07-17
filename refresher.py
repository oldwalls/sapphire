import os
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd, desc=""):
    print(f"\n🔄 {desc}...\n> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {desc}")
        sys.exit(1)
    else:
        print(f"✅ Success: {desc}")


def main():
    print("🔁 PHASERS REFRESHER INITIATED 🔁")

    # 1. Pull latest from git
    run_cmd("git pull", "Pull latest code from GitHub")

   
    print("\n🚀 Refresh complete. Ready to launch Phasers.\n")
    print("👉 You can now run: `python sapphire_chat.py` or `python phasers_cli.py`")

if __name__ == "__main__":
    main()
