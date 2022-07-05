#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""External shell interaction functions"""

from __future__ import annotations
from typing import Union
import sys
from shlex import split
import subprocess


def run_cmd(
    cmd: Union[str, list[str]],
    print_output: bool = True,
    shell: bool = False,
    check: bool = True,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Executes command in Shell.

    Args:
        cmd: Command to be executed in external shell. It could be a
            string or a list of command parts (see subprocess function
            'run' for details).
        print_output: If true, print out the shell outputs.
        shell: If true, the command will be executed through the shell
            (see subprocess doc for details).
        check: If check is true, and the process exits with a non-zero
            exit code, a CalledProcessError exception will be raised.
            Attributes of that exception hold the arguments, the exit
            code, and stdout and stderr if they were captured.
        **kwargs: Additional keyword arguments pass to function 'run'.

    Returns:
        A subprocess.CompletedProcess object.
    """

    try:
        if shell:
            if isinstance(cmd, list):
                cmd = " ".join(cmd)
            res = subprocess.run(
                cmd, shell=True, capture_output=True, check=check, encoding="utf-8", **kwargs
            )
        else:
            if isinstance(cmd, str):
                cmd = split(cmd)
            res = subprocess.run(cmd, capture_output=True, check=check, encoding="utf-8", **kwargs)
        if print_output:
            if res.stdout != "":
                print(res.stdout.rstrip("\n"), flush=True)
            if res.stderr != "":
                print(res.stderr, flush=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
    return res
