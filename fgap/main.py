import argparse

from logger_module import setup_logging, log_message
from commmand_module import *
import fgap_config
from darshan_feature_module import get_save_darshan_features

from prediction_module import pred_save_perf
from ado_module import ado

"""
[fgap_config.py]                :global configuration and vars for fgap

[main.py]                       :main() for fgap workfolw

[get_all_log_features.py]       :merge all darshan log features to one excel for model training
    >> get_all(log_dir, output_excel):
        >> 1. get darshan features from [logdir]/*.dxt.txt to excel [output_excel]
        >> 2. merge all xlsx file in the same directory only keep one headline
        
[train_write.py]
[train_read.py]                 :train read_models from [input_merge.xlsx] to models/read_xxx.joblib+read-xxx.png
        
[logger_module.py]              :logger module
    >> setup_logging():
        >> init logging
    >> log_message(str):
        >> logging [str] to console or log_file decided in fgap_config.py
    
[command_module.py]             :deal kinds of command, maybe divided into darshan_module + ...

[darshan_feature_module.py]     :get features from darshan log

[vgg_module.py]                 :get I/O features from I/O time-offset image 

[prediction_module.py]          :predict write/read performance using specified model and output to 
    >> pred_save_perf(excel_file_path, record_performance_path)
        >> get features from [excel_file_path]
        >> re-construct features with [fs.config]
        >> predict write/read performance 
        >> save to [record_performance_path]

[ado_module.py]                 :read pred.txt and output file_fs_tag

models/                         :save joblib and png

"""

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Execute a command with optional output.")
    parser.add_argument('--log-run', action='store_true',
                        help='fagp all process: executing with darshan; ado; run again.')
    parser.add_argument('--run', action='store_true',
                        help='Execute the command directly.')
    parser.add_argument('--exe', type=str, required=True,
                        help='The command to execute.')
    return parser

def validate_arguments(args):
    """Validate command line arguments to ensure --run and --log-run are not used together."""
    if args.run and args.log_run:
        print("Error: --run and --log-run cannot be used together.")
        sys.exit(1)

def main():
    """Main function coordinating the execution of various parts."""
    setup_logging()
    parser = create_parser()
    args = parser.parse_args()
    validate_arguments(args)
    log_message("fgap initialized")

    command = args.exe
    fgap_config.INPUT_COMMAND = command # save command to global var: INPUT_COMMAND
    save_num_hosts_from_command(command) # save NUM_HOSTS in fgap.config

    log_message("input command: {}" .format(command))

    """
    0. train the model 
        >> [python3 get_all_log_features.py  darshan-log/ total.xlsx]
                merge all darshan dxt log to one merged excel
        >> [python3 train_write.py total.xlsx]
        >> [python3 train_read.py total.xlsx]
                train and save models in dir models/ for write/read bw predicrion
    """

    """
    1. run with darshan dxt
        darshan_log_file = DARSHAN_LOG_DIR + basename + ".darshan"
        >> darshan_txt
        >> darshan_dxt_txt
    TODO: only support mpirun now
    """
    command = add_darshan_lib(command)
    run_with_darshan_dxt(command)
    darshan_del_log()

    """
    2. get features from darshan log
        
    """
    # fgap_config.DARSHAN_LOG_FILE = "darshan-log/4nodes-ppn1-100m-5m-F-C-g-v1.darshan" # for debug
    darshan_dxt_txt = fgap_config.DARSHAN_LOG_FILE + ".dxt.txt"
    feature_file = fgap_config.DARSHAN_LOG_FILE + ".xlsx"
    image_path = os.path.dirname(darshan_dxt_txt)
    get_save_darshan_features(darshan_dxt_txt, feature_file, image_path)
    log_message(f"darshan dxt txt: [{darshan_dxt_txt}]")
    log_message(f"feature_file: [{feature_file}]")


    """
    3. predicted performance
    
    """
    # note: the wirte/read model is specified in code
    #           we should choose the best one carefully
    pred_perf_file = fgap_config.DARSHAN_LOG_FILE + ".pred.txt"
    pred_save_perf(feature_file, pred_perf_file)
    log_message(f"predicted performance: [{pred_perf_file}]")
    """
    4. create file_tag mapping by ADO
    
    """
    fgap_config.FGAP_FILE_TAG = fgap_config.DARSHAN_LOG_FILE + ".file_fs_tag"
        ado(pred_perf_file, fgap_config.FGAP_FILE_TAG)
    log_message(f"ado: FGAP_FS: [{fgap_config.FGAP_FS}], FGAP_FILE_TAG: [{fgap_config.FGAP_FILE_TAG}]")

    """
    5. run again with file_tag and libgkfs
    
    """
    client_cmd = add_libgkfs_lib(fgap_config.INPUT_COMMAND)
    env_set = f"export FGAP_FS={fgap_config.FGAP_FS}; export FGAP_FILE_TAG={fgap_config.FGAP_FILE_TAG}; s"
    upgrade_cmd = add_envset(client_cmd, env_set)
    log_message(upgrade_cmd)
    execute_command(upgrade_cmd)


if __name__ == "__main__":
    main()
