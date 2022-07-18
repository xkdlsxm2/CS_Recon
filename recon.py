import os, json, pathlib, re, utils
from argparse import ArgumentParser

PATTERN = re.compile("^MID\d{5}_FID\d{5}(.*).h5$") # File name: MIDxxxxx_FIDxxxxx.h5 (x=int)

def recon(args):
    method = utils.choose_method(args)
    for root, dir, files in os.walk(args.data_path):
        for file in files:
            # m = PATTERN.match(file)
            m = True
            if m:
                file = pathlib.Path(root) / file # convert string to pathlib object
                if args.data_name is None:  # Processing for all dataset
                    method(file, args)
                else:  # Processing for specific dataset
                    if file.name in args.data_name:
                        method(file, args)


def build_args(config_json):
    parser = ArgumentParser()

    config = config_json['path']
    parser.add_argument(
        '--save_path',
        type=str,
        default=pathlib.Path(config["save_path"]),
        help='path to save reconstruction images and pickles')

    parser.add_argument(
        '--data_path',
        type=str,
        default=pathlib.Path(config["data_path"]),
        help='path to data')

    parser.add_argument(
        '--data_name',
        default=config["data_name"],
        help='data name to be reconstructed (None: all data in the path')

    config = config_json['recon']
    parser.add_argument(
        '--method',
        type=str,
        default=config["method"],
        help='Reconstruction method')

    parser.add_argument(
        '--rate',
        type=int,
        default=config["undersamping_rate"],
        help='Undersamping_rate')

    config = config_json['cs']
    parser.add_argument(
        '--ESPIRiT_threshold',
        type=float,
        default=config["ESPIRiT_threshold"],
        help='ESPIRiT_threshold')

    parser.add_argument(
        '--CS_lambda',
        type=float,
        default=config["CS_lambda"],
        help='CS_lambda')

    config = config_json['grappa']
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=config["kernel_size"],
        help='kernel_size for weigh estimation')

    args = parser.parse_args()
    data_path = args.data_path
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_job_id = "." if slurm_job_id == None else f"{slurm_job_id}.tinygpu"

    data_path = data_path.parent / slurm_job_id / data_path.name
    args.data_path = data_path

    return args

def run():
    config = json.load(open(pathlib.Path(__file__).parent / "config.json"))
    args = build_args(config)
    recon(args)


if __name__ == "__main__":
    run()
