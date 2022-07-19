import os, json, pathlib, re, utils

PATTERN = re.compile("MID\d{5}_FID\d{5}.h5") # File name: MIDxxxxx_FIDxxxxx.h5 (x=int)

def recon(args, method):
    method = utils.choose_method(method)
    for root, dir, files in os.walk(args.data_path):
        for file in files:
            m = PATTERN.match(file)
            if m:
                file = pathlib.Path(root) / file # convert string to pathlib object
                if args.data_name is None:  # Processing for all dataset
                    method(file, args)
                else:  # Processing for specific dataset
                    if file.name in args.data_name:
                        method(file, args)

def run():
    config = json.load(open(pathlib.Path(__file__).parent / "config.json"))
    args = utils.build_args(config)
    for method in args.method:
        for rate in args.rates:
            args.rate = rate
            recon(args, method)


if __name__ == "__main__":
    run()
