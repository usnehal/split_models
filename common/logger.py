from common.constants import bcolors
from common.config import Config

debug_level = 0
cfg = Config()

class Logger:
    def __init__(self):
        debugLogs = False
        debug_level = 1
    def set_log_level(level):
        global debug_level
        debug_level = level

    def get_log_level(level):
        return debug_level

    def debug_print(str):
        if(debug_level >= 2):
            print(str)

    def event_print(str):
        if(debug_level >= 1):
            print(bcolors.OKCYAN + str + bcolors.ENDC)

    def milestone_print(str):
        if(debug_level >= 0):
            print(bcolors.OKGREEN + str + bcolors.ENDC)
            file1 = open(cfg.temp_path + "/log.txt", "a")  # append mode
            file1.write(str + "\n")
            file1.close()