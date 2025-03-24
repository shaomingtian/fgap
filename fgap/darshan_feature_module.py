import re
import os
import argparse
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from logger_module import log_message
from vgg_module import get_image_feature_to_excel
DARSHAN_LOGFILE=''

class ExeInfo:
    def __init__(self, app, paras, nprocs, run_time):
        self.app = app
        self.paras = paras
        self.nprocs = nprocs
        self.run_time = run_time

def get_exe_info(log_text):
    # info to get
    exe_re = re.compile(r"# exe: (.+)")
    nprocs_re = re.compile(r"# nprocs: (\d+)")
    run_time_re = re.compile(r"# run time: ([\d\.]+)")

    app = None
    paras = None
    nprocs = None
    run_time = None

    # read lines
    lines = log_text.strip().split('\n')

    for line in lines:
        # search exe line
        exe_match = exe_re.search(line)
        if exe_match:
            exe_string = exe_match.group(1).strip()
            parts = exe_string.split()
            app = parts[0]  # 第一个部分为 app
            paras = ' '.join(parts[1:])  # 剩余部分为 paras

        # search nprocs line
        nprocs_match = nprocs_re.search(line)
        if nprocs_match:
            nprocs = int(nprocs_match.group(1))

        # search run time line
        run_time_match = run_time_re.search(line)
        if run_time_match:
            run_time = float(run_time_match.group(1))

    return ExeInfo(app, paras, nprocs, run_time)

def get_file_info(log_text):
    # Regular expressions to find relevant data
    file_info_re = re.compile(r"# DXT, file_id: \d+, file_name: (?P<file_name>.+)")
    rank_info_re = re.compile(r"# DXT, rank: (?P<rank>\d+), hostname: (?P<hostname>.+)")
    fs_info_re = re.compile(r"# DXT, mnt_pt: (?P<mnt_pt>.+), fs_type: (?P<fs_type>\S+)")
    stripe_info_re = re.compile(r"#\s*\[Component 1\]\s*stripe_ext:\s*0\s*-\s*EOF,\s*stripe_size:\s*(?P<stripe_size>\d+),\s*stripe_count:\s*(?P<stripe_count>\d+),\s*OSTs:\s*(?P<osts>\d+)")
    #stripe_info_re = re.compile(r"# DXT, Lustre stripe_size: (?P<stripe_size>\d+), Lustre stripe_count: (?P<stripe_count>\d+)")
    #stripe_info_re = re.compile(r"#       \[Component 1\] stripe_ext: 0 - EOF, stripe_size: (?P<stripe_size>\d+), stripe_count: (?P<stripe_count>\d+), OSTs: (?P<osts>\d+)")
    #stripe_info_re = re.compile(r"#       \[Component 1\] stripe_ext: 0 - EOF, stripe_size: (?P<stripe_size>\d+), stripe_count: (?P<stripe_count>\d+), OSTs: (?P<osts>\d+)")
    data_re = re.compile(r"^\s*(?P<module>\S+)\s+(?P<rank>\d+)\s+(?P<wt_rd>\S+)\s+(?P<segment>\d+)\s+(?P<offset>\d+)\s+(?P<length>\d+)\s+(?P<start>\d+\.\d+)\s+(?P<end>\d+\.\d+)\s+\[\s*(?P<ost>\d+)\s*\]")

    # Data structure to store the parsed information
    parsed_data = defaultdict(lambda: {
        'rank_info': [],
        'write': [],
        'read': [],
        'modules': set(),
        'fs': {},
        'size': 0,
        'num_rank': 0,
        'num_hostname': 0,
        'write_bw': 0.0,
        'read_bw': 0.0,
        'top_write_length': 0,
        'top_read_length': 0,
        'time_start_access': -1, # min start
        'time_end_access': -1,  # max end
        'total_write_size': -1,
        'total_read_size': -1
    })

    current_file_name = None
    current_fs_type = None

    # Split log into lines
    lines = log_text.strip().split('\n')
    current_rank = None
    current_hostname = None

    for line in lines:
        # Check for file_name and rank information
        file_match = file_info_re.search(line)
        if file_match:
            current_file_name = file_match.group("file_name")
            continue

        rank_match = rank_info_re.search(line)
        if rank_match and current_file_name:
            current_rank = rank_match.group("rank")
            current_hostname = rank_match.group("hostname")
            parsed_data[current_file_name]['rank_info'].append({
                'rank': current_rank,
                'hostname': current_hostname
            })
            continue

        # Match file system type
        fs_match = fs_info_re.search(line)
        if fs_match and current_file_name:
            current_fs_type = fs_match.group("fs_type")
            parsed_data[current_file_name]['fs']['type'] = current_fs_type
            continue

        # Match stripe size and stripe count info
        if current_fs_type == 'lustre':
            stripe_match = stripe_info_re.search(line)
            if stripe_match and current_file_name:
                parsed_data[current_file_name]['fs']['stripe_size'] = stripe_match.group("stripe_size")
                parsed_data[current_file_name]['fs']['stripe_count'] = stripe_match.group("stripe_count")

        # Match data lines for write and read
        data_match = data_re.match(line)
        if data_match and current_file_name:
            module = data_match.group("module")
            wt_rd = data_match.group("wt_rd")
            segment = data_match.group("segment")
            offset = int(data_match.group("offset"))
            length = int(data_match.group("length"))
            start = float(data_match.group("start"))
            end = float(data_match.group("end"))

            parsed_data[current_file_name]['modules'].add(module)

            segment_end = offset + length

            if segment_end > parsed_data[current_file_name]['size']:
                parsed_data[current_file_name]['size'] = segment_end

            if wt_rd == "write":
                parsed_data[current_file_name]['write'].append({
                    'rank': current_rank,
                    'hostname': current_hostname,
                    'segment': segment,
                    'offset': offset,
                    'length': length,
                    'start': start,
                    'end': end
                })
            elif wt_rd == "read":
                parsed_data[current_file_name]['read'].append({
                    'rank': current_rank,
                    'hostname': current_hostname,
                    'segment': segment,
                    'offset': offset,
                    'length': length,
                    'start': start,
                    'end': end
                })

    # Calculate additional statistics
    for file_name, data in parsed_data.items():
        unique_ranks = {info['rank'] for info in data['rank_info']}
        unique_hostnames = {info['hostname'] for info in data['rank_info']}

        data['num_rank'] = len(unique_ranks)
        data['num_hostname'] = len(unique_hostnames)
        
        data['time_start_access'] = min(min(write['start'] for write in data['write']), min(read['start'] for read in data['read']))
        data['time_end_access'] = max(max(write['end'] for write in data['write']), max(read['end'] for read in data['read']))

        total_write_length = sum(write['length'] for write in data['write'])
        data['total_write_size'] = total_write_length
        if data['write']:
            total_write_start = min(write['start'] for write in data['write'])
            total_write_end = max(write['end'] for write in data['write'])
            total_write_time = total_write_end - total_write_start
            if total_write_time > 0:
                data['write_bw'] = (total_write_length / total_write_time) / (1024 ** 2)

            write_lengths = [write['length'] for write in data['write']]
            if write_lengths:
                length_counts = Counter(write_lengths)
                data['top_write_length'] = length_counts.most_common(1)[0][0]

        total_read_length = sum(read['length'] for read in data['read'])
        data['total_read_size'] = total_read_length
        if data['read']:
            total_read_start = min(read['start'] for read in data['read'])
            total_read_end = max(read['end'] for read in data['read'])
            total_read_time = total_read_end - total_read_start
            if total_read_time > 0:
                data['read_bw'] = (total_read_length / total_read_time) / (1024 ** 2)

            read_lengths = [read['length'] for read in data['read']]
            if read_lengths:
                length_counts = Counter(read_lengths)
                data['top_read_length'] = length_counts.most_common(1)[0][0]

        data['write_bw'] = round(data['write_bw'], 2)
        data['read_bw'] = round(data['read_bw'], 2)
        data['modules'] = list(data['modules'])

    return parsed_data

# Function to save parsed data to an Excel file
def save_to_excel(parsed_data, output_file):
    global DARSHAN_LOGFILE
    # Prepare data for DataFrame
    file_info, exe_info = parsed_data
    data_to_save = []
    num_files = len(file_info)


    for file_name, data in file_info.items():
        row = {
            'FileName': file_name, # not for train
            'basename': os.path.basename(file_name), # not for train
            'darshan_log': os.path.basename(DARSHAN_LOGFILE), # not for train
            'exe': exe_info.app,
            'ExeParameters': exe_info.paras,
            'ExeNumProcs': exe_info.nprocs,
            'ExeNumFiles': num_files,
            'FileSize': data['size'],
            'IOmethod': ','.join(data['modules']),  # Join module names with commas
            'FileSystem': data['fs']['type'],
            'StripeSize': data['fs']['stripe_size'],
            'StripeCount': data['fs']['stripe_count'],
            'FileNumRanks': data['num_rank'],
            'FileNumHosts': data['num_hostname'],
            'TopWriteLength': data['top_write_length'],
            'TopReadLength': data['top_read_length'],
            'WriteBw': data['write_bw'],
            'ReadBw': data['read_bw'],
            'time_start_access': data['time_start_access'], # not for train
            'time_end_access': data['time_end_access'],  # not for train
            'total_write_size': data['total_write_size'], # not for train
            'total_read_size': data['total_read_size'] # not for train
        }
        data_to_save.append(row)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(data_to_save)
    df.to_excel(output_file, index=False)

# Read log file and parse it
def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        log_text = file.read()
    return get_file_info(log_text), get_exe_info(log_text)

# Function to draw rectangles for each file's accesses
def plot_accesses(file_info, output_directory):
    colors = {}
    color_index = 0

    for file_name, data in file_info.items():
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # 先绘制写入矩形
        for record in data['write']:
            hostname = record['hostname']
            if hostname not in colors:
                colors[hostname] = plt.cm.tab10(color_index)
                color_index += 1

            plt.gca().add_patch(plt.Rectangle(
                (record['start'], record['offset']),
                record['end'] - record['start'],
                record['length'],
                color=colors[hostname],
                alpha=0.5,
                label=hostname + ' Write' if hostname + ' Write' not in ax.get_legend_handles_labels()[1] else ""
            ))

        # 接下来绘制读取矩形
        for record in data['read']:
            hostname = record['hostname']
            if hostname not in colors:
                colors[hostname] = plt.cm.tab10(color_index)
                color_index += 1

            plt.gca().add_patch(plt.Rectangle(
                (record['start'], record['offset']),
                record['end'] - record['start'],
                record['length'],
                color=colors[hostname],
                alpha=0.5,
                label=hostname + ' Read' if hostname + ' Read' not in ax.get_legend_handles_labels()[1] else ""
            ))

        ax.autoscale()
        # plt.title(f'Access Pattern for {file_name}')
        # plt.xlabel('Time')
        # plt.ylabel('Offset')

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = set(labels)
        # if len(unique_labels) > 0:
        #     ax.legend(handles, unique_labels, loc='upper right')

        # plt.grid()
        pngname = os.path.basename(file_name)
        plt.savefig(f"{output_directory}/{pngname}.png")
        plt.close()





def get_save_darshan_features(log_file_path, output_file_path, image_path):
    global DARSHAN_LOGFILE
    DARSHAN_LOGFILE = log_file_path

    # Parse the log file
    parsed_result = parse_log_file(log_file_path)

    # Save results to Excel
    save_to_excel(parsed_result, output_file_path)

    file_info, exe_info = parsed_result

    # draw images
    plot_accesses(file_info, image_path)

    for file_name, data in file_info.items():
        get_image_feature_to_excel(os.path.dirname(DARSHAN_LOGFILE) + "/" + os.path.basename(file_name) + ".png", log_file_path.rstrip(".dxt.txt") + ".xlsx")



    # Output results to console as well
    for file_name, data in file_info.items():
        print(f"File: {file_name}")
        print(f"Rank Info: {data['rank_info']}")
        # print(f"Write Data: {data['write']}")
        # print(f"Read Data: {data['read']}")
        print(f"Modules: {','.join(data['modules'])}")  # Join module names with commas
        print(f"File System Info: {data['fs']['type']}")
        print(f"File Size: {data['size']}")
        print(f"Number of Unique Ranks: {data['num_rank']}")
        print(f"Number of Unique Hostnames: {data['num_hostname']}")
        print(f"Write Bandwidth (write_bw): {data['write_bw']} MB/s")
        print(f"Read Bandwidth (read_bw): {data['read_bw']} MB/s")
        print(f"Top Write Length: {data['top_write_length']}")
        print(f"Top Read Length: {data['top_read_length']}")
        print("")
    print(f"App: {exe_info.app}")
    print(f"Parameters: {exe_info.paras}")
    print(f"Number of Processes: {exe_info.nprocs}")
    print(f"Run Time: {exe_info.run_time}")


def get_save_darshan_features(log_file_path, output_file_path, image_path):
    global DARSHAN_LOGFILE
    DARSHAN_LOGFILE = log_file_path

    # Parse the log file
    parsed_result = parse_log_file(log_file_path)

    # Save results to Excel
    save_to_excel(parsed_result, output_file_path)

    file_info, exe_info = parsed_result

    # draw images
    plot_accesses(file_info, image_path)

    for file_name, data in file_info.items():
        get_image_feature_to_excel(os.path.dirname(DARSHAN_LOGFILE) + "/" + os.path.basename(file_name) + ".png", log_file_path.rstrip(".dxt.txt") + ".xlsx")

# get the date for model training
# not used now
def batch_process_log_files(log_file_paths, output_file_path):
    all_data_to_save = []  # 用于存储所有文件数据的列表

    # for filename in os.listdir(args.input):
    #     if filename.endswith('.darshan.dxt.txt'):
    for log_file_path in log_file_paths:
        try:
            parsed_result = parse_log_file(log_file_path)
            file_info, exe_info = parsed_result

            for file_name, data in file_info.items():
                row = {
                    'FileName': file_name,
                    'basename': os.path.basename(file_name),
                    'darshan_log': os.path.basename(log_file_path),
                    'exe': exe_info.app,
                    'ExeParameters': exe_info.paras,
                    'ExeNumProcs': exe_info.nprocs,
                    'FileSize': data['size'],
                    'IOmethod': ','.join(data['modules']),
                    'FileSystem': data['fs']['type'],
                    'StripeSize': data['fs'].get('stripe_size', 'N/A'),
                    'StripeCount': data['fs'].get('stripe_count', 'N/A'),
                    'FileNumRanks': data['num_rank'],
                    'FileNumHosts': data['num_hostname'],
                    'TopWriteLength': data['top_write_length'],
                    'TopReadLength': data['top_read_length'],
                    'WriteBw': data['write_bw'],
                    'ReadBw': data['read_bw'],
                }
                all_data_to_save.append(row)
        except Exception as e:
            log_message(f"Error processing {log_file_path}: {e}")

    # 创建 DataFrame 并保存到 Excel
    df = pd.DataFrame(all_data_to_save)
    df.to_excel(output_file_path, index=False)

    log_message(f"Successfully saved all data to {output_file_path}")


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse a log file and extract information.')
    parser.add_argument('log_file', type=str, help='Path to the log file to parse')
    parser.add_argument('output_file', type=str, help='Path to save the output Excel file')
    parser.add_argument('output_dir', type=str, help='Directory to save the output plots')



    args = parser.parse_args()
    log_file_path = args.log_file
    global DARSHAN_LOGFILE
    DARSHAN_LOGFILE = log_file_path
    output_file_path = args.output_file

    output_directory = args.output_dir

    # Parse the log file
    parsed_result = parse_log_file(log_file_path)

    # Save results to Excel
    save_to_excel(parsed_result, output_file_path)




    file_info, exe_info = parsed_result
    plot_accesses(file_info, output_directory)

    # Output results to console as well
    for file_name, data in file_info.items():
        print(f"File: {file_name}")
        print(f"Rank Info: {data['rank_info']}")
        print(f"Write Data: {data['write']}")
        print(f"Read Data: {data['read']}")
        print(f"Modules: {','.join(data['modules'])}")  # Join module names with commas
        print(f"File System Info: {data['fs']['type']}")
        print(f"File Size: {data['size']}")
        print(f"Number of Unique Ranks: {data['num_rank']}")
        print(f"Number of Unique Hostnames: {data['num_hostname']}")
        print(f"Write Bandwidth (write_bw): {data['write_bw']} MB/s")
        print(f"Read Bandwidth (read_bw): {data['read_bw']} MB/s")
        print(f"Top Write Length: {data['top_write_length']}")
        print(f"Top Read Length: {data['top_read_length']}")
        print("")
    print(f"App: {exe_info.app}")
    print(f"Parameters: {exe_info.paras}")
    print(f"Number of Processes: {exe_info.nprocs}")
    print(f"Run Time: {exe_info.run_time}")

if __name__ == "__main__":
    main()
