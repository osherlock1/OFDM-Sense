from pathlib import Path
import argparse
from tqdm import tqdm

from ofdm.processing import pipeline
from ofdm.config import OFDMConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_pth", type=str, required=True, help="path to experiment directory containing archive subdirectories")
    parser.add_argument("--ref_pth", type=str, default="./data_files/rand_ofdm_packet_ref.json")
    args = parser.parse_args()

    ofdm_conf = OFDMConfig()
    experiment_pth = Path(args.experiment_pth)
    for archive_pth in experiment_pth.glob("*archive"):
        print(f"\n--- Processing {archive_pth} ---")
        pipeline.process_archive(archive_dir=archive_pth, ref_path=args.ref_pth, ofdm_conf=ofdm_conf)
    
if __name__ == "__main__":
    main()