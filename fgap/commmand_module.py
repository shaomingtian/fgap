import subprocess
import sys
import os
from datetime import datetime
from ClusterShell.NodeSet import NodeSet

from logger_module import log_message
from fgap_config import DARSHAN_LIB, GEKKOFS_LIB, DARSHAN_LOG_DIR
import fgap_config

def execute_command(command):
    """Execute the command and return its output in real-time."""
    try:
        log_message("execve [{}]".format(command))

        # Use Popen to execute the command
        process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Read output line by line in real-time
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end='')  # Print each line as it is produced
            # log_message(stdout_line.strip())  # Log the output if needed

        process.stdout.close()  # Close the stdout pipe
        process.wait()  # Wait for the process to finish

        # Check if the process exited with status code 0
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: [{command}] with stderr: {e.stderr}")
        sys.exit(1)

def replace_spaces_with_underscores(input_string):
    return input_string.replace(' ', '_')

def replace_dot_with_underscores(input_string):
    return input_string.replace(',', '_')

def replace_gang_with_underscores(input_string):
    return input_string.replace('/', '_')

def truncate_string_within200(text):
    """the file name is limited to 255 char"""
    if len(text) > 200:
        truncated_text = text[:200]
        return truncated_text
    else:
        return text


def get_hosts_from_mpirun(command):
    # Split the command string into parts
    parts = command.split()

    # Iterate through all parts to find the -hosts option
    for i in range(len(parts)):
        if parts[i] == '-hosts' and i + 1 < len(parts):
            # Return the parameter after -hosts
            return parts[i + 1]

    # If -hosts is not found, return None or an appropriate message
    return None

def save_num_hosts_from_command(command):
    if command.count("mpirun") > 0:
        # mpirun -hosts
        hosts = get_hosts_from_mpirun(command)
        num_hosts = len(NodeSet(hosts))
    elif command.count("srun") > 0 or command.count("yhrun") > 0:
        # TODO
        num_hosts = ""
    fgap_config.NUM_HOSTS = num_hosts


def add_darshan_lib(mpirun_command):
    """Add '-env xxx' option to the given mpirun command."""
    # Splitting the command to find the position of 'mpirun'
    parts = mpirun_command.split()

    # Check if the command starts with 'mpirun'
    if parts[0] == "mpirun":
        # Insert the '-env' option right after 'mpirun'
        parts.insert(1, f"-env LD_PRELOAD={DARSHAN_LIB}")

        # Join the parts back into a single command string
        return " ".join(parts)
    else:
        raise ValueError("The command does not start with 'mpirun'.")


def add_libgkfs_lib(mpirun_command):
    """Add '-env xxx' option to the given mpirun command."""
    # Splitting the command to find the position of 'mpirun'
    parts = mpirun_command.split()

    # Check if the command starts with 'mpirun'
    if parts[0] == "mpirun":
        # Insert the '-env' option right after 'mpirun'
        parts.insert(1, f"-env LD_PRELOAD={GEKKOFS_LIB}")

        # Join the parts back into a single command string
        return " ".join(parts)
    else:
        raise ValueError("The command does not start with 'mpirun'.")


def add_envset(command, env_set_str):
    """Add export xxx=xxx; before the commmand"""
    return env_set_str + " " + command

def set_darshan_log_file():
    """set darshan dxt env"""
    print("input: {}" .format(fgap_config.INPUT_COMMAND))
    basename = replace_spaces_with_underscores(fgap_config.INPUT_COMMAND)
    basename = replace_dot_with_underscores(basename)
    basename = replace_gang_with_underscores(basename)
    basename = truncate_string_within200(basename)
    global DARSHAN_LOG_DIR
    if not fgap_config.DARSHAN_LOG_DIR.endswith('/'):
        fgap_config.DARSHAN_LOG_DIR += '/'

    now = datetime.now()
    directory_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    fgap_config.DARSHAN_LOG_DIR += directory_name + "/"
    fgap_config.DARSHAN_LOG_FILE = fgap_config.DARSHAN_LOG_DIR + basename + ".darshan"

def set_darshan_dxt_config(command):
    darshan_log_file = fgap_config.DARSHAN_LOG_FILE
    if not os.path.exists(fgap_config.DARSHAN_LOG_DIR):
        """if DARSHAN_LOG_DIR doesn't exist, create it first"""
        try:
            os.makedirs(fgap_config.DARSHAN_LOG_DIR)
        except Exception as e:
            print(f"error when mkdir {fgap_config.DARSHAN_LOG_DIR}: {e}")
    env = "export DARSHAN_LOGFILE=" +  darshan_log_file + "; "
    # env += "export DARSHAN_ENABLE_NONMPI=1; "
    env += "export DXT_ENABLE_IO_TRACE=1; "
    return add_envset(command, env)

def run_with_darshan_dxt(command):
    """ set darshan dxt env and run with libdarshan.so"""
    set_darshan_log_file()
    command = set_darshan_dxt_config(command)
    execute_command(command)

def darshan_del_log():
    darshan_txt = fgap_config.DARSHAN_LOG_FILE + ".txt"
    darshan_command = "{} --all {} >> {}" .format(fgap_config.DARSHAN_BIN,
                                                  fgap_config.DARSHAN_LOG_FILE, darshan_txt)
    execute_command(darshan_command)

    darshan_dxt_txt = fgap_config.DARSHAN_LOG_FILE + ".dxt.txt"
    darshan_dxt_command = "{} {} >> {}" .format(fgap_config.DARSHAN_DXT_BIN,
                                                  fgap_config.DARSHAN_LOG_FILE, darshan_dxt_txt)
    execute_command(darshan_dxt_command)


