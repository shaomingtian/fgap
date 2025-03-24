import argparse
import pandas as pd

from logger_module import setup_logging, log_message
from commmand_module import *
import fgap_config
from darshan_feature_module import get_save_darshan_features




def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Get all features from a directory to a merged excel.")
    parser.add_argument('log_dir', type=str, help='Path to the log file directory to parse')
    parser.add_argument('output_excel', type=str, help='Path to save the output Excel file')

    return parser



def merge_excel_files(directory, output_file):
    """
    merge all xlsx file in the same directory
    only keep one headline

    :param directory: the dir containing .xlsx
    :param output_file: merged xlsx file
    """

    all_data = []
    # find all xlsx file in dir
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
                path = os.path.join(root, file)
                df = pd.read_excel(path)
                all_data.append(df)

    # merge all DataFrame anb keep one headline
    combined_data = pd.concat(all_data, ignore_index=True)

    # write merged data to output file
    combined_data.to_excel(output_file, index=False)


def get_all(log_path, output_excel):

    dxt_txt_files = []

    # os.walk() to scan all dir/subdir
    for root, dirs, files in os.walk(log_path):
        for file in files:
            # check if end with .txt
            if file.endswith('.dxt.txt'):
                # get the file path
                path = os.path.join(root, file)
                dxt_txt_files.append(path)


    for dxt_txt_file in dxt_txt_files:
        feature_file = dxt_txt_file[:-8] + ".xlsx"
        image_path = os.path.dirname(dxt_txt_file)
        print(dxt_txt_file, feature_file, image_path) # for debug
        get_save_darshan_features(dxt_txt_file, feature_file, image_path)

    merge_excel_files(log_path, output_excel)


def main():
    """
    1. get darshan features from [logdir]/*.dxt.txt to excel [output_excel]
    2. merge all xlsx file in the same directory
        only keep one headline

    :param log_dir: the dir containing .xlsx
    :param output_excel: merged xlsx file
    """
    parser = create_parser()
    args = parser.parse_args()
    get_all(args.log_dir, args.output_excel)



if __name__ == "__main__":
    main()
