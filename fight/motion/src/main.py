import argparse
from core.config import load_config
from service.motion_service import run_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/motion.yaml")
    ap.add_argument("--source", required=True, help="RTSP url or video file path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_motion(args.source, cfg)


if __name__ == "__main__":
    main()