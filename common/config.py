import time
import socket

class Config():
    def __init__(self, server_ip=None):
        self.host = None
        self.HOST_NIMBLE = 1
        self.HOST_WSL = 2
        self.HOST_PI = 3

        self.total_test_images = 5000
        self.max_tokenized_words = 20000
        self.MAX_SEQ_LENGTH = 25
        self.batch_size = 32
        self.embedding_dim = 256 
        self.units = 512
        if(server_ip != None):
            self.server_ip = server_ip
        else:
            self.server_ip = 'localhost'
        # self.server_ip = '35.200.232.85'
        self.server_port = 5001

        host = socket.gethostname()
        self.workspace_path = '/home/suphale/WorkSpace' 
        self.images_path = None
        if('cuda' in host):
            self.host = self.HOST_NIMBLE
            self.workspace_path='/mnt/disks/user/project/WorkSpace'
            # print("In NimbleBox")
        if(host == 'LTsuphale-NC2JM'):
            self.host = self.HOST_WSL
            # print("In WSL")
        if(host == 'raspberrypi'):
            self.host = self.HOST_PI
            self.workspace_path='/home/pi/WorkSpace'
            # print("In raspberry-pi")
        self.images_path = self.workspace_path + '/test_images'
        self.saved_model_path = self.workspace_path + '/saved_model'
        self.temp_path = self.workspace_path + '/temp'

        self.text_file = self.workspace_path + '/lists/captions_' + str(self.total_test_images) + '.txt'
        self.list_file = self.workspace_path + '/lists/images_' + str(self.total_test_images) + '.txt'
        self.instances_file = self.workspace_path + '/lists/instances_' + str(self.total_test_images) + '.txt'
        self.df_captions_csv = self.workspace_path + '/lists/df_captions_' + str(self.total_test_images) + '.csv'
        self.df_instances_csv = self.workspace_path + '/lists/df_instances_' + str(self.total_test_images) + '.csv'
        self.df_categories_csv = self.workspace_path + '/lists/df_categories.csv'

        self.timestr = time.strftime("%Y_%m_%d-%H_%M")

        self.PREDICTIONS_THRESHOLD = 0.4

    def __getitem__(self):

        return self.x

    def __len__(self):
        return len(self.x)