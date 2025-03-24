# Define macros for logging options
INPUT_COMMAND="" # global var: save the input command
NUM_HOSTS="" # global var : save the number of nodes occupied by this task


LOG_TO_FILE = True  # Set to True to output to a file
LOG_TO_CONSOLE = True  # Set to True to output to console
LOG_FILE_NAME = '/tmp/fgap.log'  # Specify log file name

DARSHAN_LIB="/thfs3/home/wuhuijun/smt/fgap/install/darshan/lib/libdarshan.so"
DARSHAN_BIN="/thfs3/home/wuhuijun/smt/fgap/install/darshan/bin/darshan-parser"
DARSHAN_DXT_BIN="/thfs3/home/wuhuijun/smt/fgap/install/darshan/bin/darshan-dxt-parser"
DARSHAN_LOG_DIR="/thfs3/home/wuhuijun/fgap/darshan_log" # global var: save the log path of this task
DARSHAN_LOG_FILE="" # global var: save the darshan log file abs path

GEKKOFS_LIB="/thfs3/home/wuhuijun/smt/gekkofs092/gekkofs-bloom/gekkofs-v0.9.2/build/src/client/libgkfs_intercept.so"
GEKKOFS_HOSTS_FILE="/thfs3/home/wuhuijun/smt/gekkofs092/gekkofs-bloom/gekkofs-v0.9.2/scripts/yh-run/gkfs_hosts.txt"

IMAGE_KEEP = False # delete the image after getting features
FGAP_FEATURE_FILE="" # global var: save the xls/csv file with I/O features
FGAP_PRE_FILE="" # global var: save the predicted performance of each file in different filesystems
FGAP_FS="/thfs3/home/wuhuijun/smt/fgap/code/final/client_fs_config" # global var: save the available backend filesystems
FGAP_FILE_TAG="" # global var: save the file tags created by ADO

FGAP_WRITE_MODEL=["models/write_model_Decision_Tree.joblib",
             "models/write_model_Extra_Trees.joblib",
             "models/write_model_Extreme_Gradient_Boosting.joblib",
             "models/write_model_Gradient_Boosting.joblib",
             "models/write_model_Histogram-based_Gradient_Boosting.joblib",
             "models/write_model_Linear_Regression.joblib",
             "models/write_model_Random_Forest.joblib"
             ]
FGAP_READ_MODEL=["models/read_model_Decision_Tree.joblib",
                  "models/read_model_Extra_Trees.joblib",
                  "models/read_model_Extreme_Gradient_Boosting.joblib",
                  "models/read_model_Gradient_Boosting.joblib",
                  "models/read_model_Histogram-based_Gradient_Boosting.joblib",
                  "models/read_model_Linear_Regression.joblib",
                  "models/read_model_Random_Forest.joblib"
                  ]

FS_CONFIG_PATH="/thfs3/home/wuhuijun/smt/fgap/code/final/fs.config"
