#!/usr/bin/env python

"""
Performance comparison CPU vs GPU.

Executor.
"""


import argparse
import os
import subprocess
import impl.utils as utils


TITLE = 'Performance comparison CPU vs GPU.'

REL_PATH_NUMSIM_EXEC = '../build/main'
REL_PATH_TMP         = './tmp'
REL_PATH_DATA        = './data'     # can be overwritten through cmdline args

DATAFILE_PREFIX = './lid-driven-cavity'
GEOM_SUFFIX     = '.geom'
PARAMS_SUFFIX   = '.params'
JSON_SUFFIX     = '.json'

GEOM_CAVITY_SIZES = [32, 64, 100, 128, 200, 256, 300]


# |---------|
# | general |
# |---------|
def abs_path_tmp():
    return os.path.dirname(os.path.abspath(__file__)) + '/' + REL_PATH_TMP

def abs_path():
    return os.path.dirname(os.path.abspath(__file__))


# |------|
# | data |
# |------|
def json_filename(cavity_size):
    return DATAFILE_PREFIX + "_{n}x{n}".format(n=cavity_size) + JSON_SUFFIX


# |------|
# | geom |
# |------|
def abs_path_geom(cavity_size):
    return abs_path_tmp() + '/' + DATAFILE_PREFIX + "_{n}x{n}".format(n=cavity_size) + GEOM_SUFFIX


def create_geom(cavity_size):
    content  = "size = {n} {n}\n".format(n=cavity_size)
    content += "length = 1.0 1.0\n"
    content += "velocity = 1.0 0.0\n"
    content += "pressure = 0.1\n"
    content += "geometry = free\n"
    content += "H" * cavity_size + "\n"
    for _ in range(0, cavity_size - 2):
        content += "#"
        content += " " * (cavity_size - 2)
        content += "#\n"
    content += "#" * cavity_size + "\n"

    with open(abs_path_geom(cavity_size), 'w') as geom:
        geom.write(content)


# |--------|
# | params |
# |--------|
def abs_path_params():
    return abs_path_tmp() + '/' + DATAFILE_PREFIX + PARAMS_SUFFIX


def create_params():
    content  = "re = 1000.0\n"
    content += "omg = 1.7\n"
    content += "alpha = 0.9\n"
    content += "dt = 0.02\n"
    content += "tend = 100.0\n"
    content += "iter = 100\n"
    content += "eps = 0.001\n"
    content += "tau = 0.5\n"

    with open(abs_path_params(), 'w') as params:
        params.write(content)


# |-----------|
# | execution |
# |-----------|
def exec_sim(abs_path_data, cavity_size):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    path_to_data  = path_to_script + '/' + REL_PATH_TMP
    stdout = path_to_script + '/' + REL_PATH_TMP + "/.tmp.{}.stdout".format(cavity_size)

    # setup execution command
    cmd = '"' + path_to_script + '/' + REL_PATH_NUMSIM_EXEC + '"'
    cmd += " --geom {}".format(abs_path_geom(cavity_size))
    cmd += " --params {}".format(abs_path_params())
    cmd += " --json {}".format(abs_path_data + '/' + json_filename(cavity_size))

    # execute
    with open(stdout, 'w') as logfile:
        subprocess.check_call(cmd, shell=True, stdout=logfile)

    os.remove(stdout)


def execute(abs_path_data):
    print(">> creating parameters for lid-driven cavity")
    create_params()
    print()

    # create geometry, execute and remove geometry for every cavity size
    for n in GEOM_CAVITY_SIZES:

        print(">> creating geometry")
        print("   lid-driven cavity ({n}x{n})".format(n=n))
        create_geom(n)
        print(">> executing")
        exec_sim(abs_path_data, n)
        print(">> removing geometry")
        print("   lid-driven cavity ({n}x{n})".format(n=n))
        os.remove(abs_path_geom(n))
        print()

    print(">> removing parameters for lid-driven cavity")
    os.remove(abs_path_params())


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument('-j', '--json', metavar='json', nargs='?', type=str, default=REL_PATH_DATA,
                        help="JSON output path relative to this py-script's location")
    args = parser.parse_args()
    abs_path_data = abs_path() + '/' + args.json

    print(">> creating directory 'tmp'")
    print()
    utils.mkdir(abs_path_tmp())

    print(">> creating directory '{}'".format(args.json))
    print()
    utils.mkdir(abs_path_data)
    execute(abs_path_data)
    print()
    print(">> removing directory 'tmp'")
    os.rmdir(abs_path_tmp())


if __name__ == '__main__':
    main()