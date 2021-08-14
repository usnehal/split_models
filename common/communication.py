import socket
import json
import zlib
import threading

from common.logger import Logger

class Client:
    def __init__(self,cfg):
        self.cfg = cfg
        # self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    def connect(self):
        Logger.debug_print("connect:Entry")
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        Logger.debug_print("connect:Connect")
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))

    def disconnect(self):
        Logger.debug_print("disconnect:Entry")
        self.s.shutdown(socket.SHUT_RDWR)
        Logger.debug_print("disconnect:Connect")
        self.s.close()

    def reconnect(self):
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.s.connect((self.cfg.server_ip,int(self.cfg.server_port)))

    def send_load_model_request(self,data_info):
        self.connect()
        Logger.debug_print("send_data:send data_info")
        self.s.send(data_info.encode())

        Logger.debug_print("send_data:receive response")
        confirmation = self.s.recv(1024).decode()
        Logger.debug_print("send_data:confirmation = "+confirmation)
        if confirmation == "OK":
            Logger.debug_print('send_data:Sending data')
            return "OK"
        else:
            print("Received error from server, %s" % (confirmation.decode()))
            return "Error"

    def send_data(self,data_info, data_buffer):
        self.connect()
        Logger.debug_print("send_data:send data_info")
        self.s.send(data_info.encode())

        Logger.debug_print("send_data:receive response")
        confirmation = self.s.recv(1024).decode()
        Logger.debug_print("send_data:confirmation = "+confirmation)
        if confirmation == "OK":
            Logger.debug_print('send_data:Sending data')
            self.s.sendall(data_buffer)

            Logger.debug_print('send_data:successfully sent data.')

            pred_caption = self.s.recv(1024)

            Logger.debug_print('send_data:received '+pred_caption.decode())
            self.s.shutdown(socket.SHUT_RDWR)
            self.s.close()
            Logger.debug_print(pred_caption.decode())
            return pred_caption.decode()
            # self.reconnect()
        else:
            print("Received error from server, %s" % (confirmation.decode()))

class Server:
    def __init__(self,cfg):
        self.cfg = cfg
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.request_count = 0
        self.callbacks = {}

        # self.accept_connections()

    def register_callback(self, obj, callback):
        # print('register_callback obj='+obj)
        if obj not in self.callbacks:
            self.callbacks[obj] = None
        self.callbacks[obj] = callback
        # print('register_callback self.callbacks=%s' % (str(self.callbacks)))
    
    def accept_connections(self):
        ip = '' 
        port = self.cfg.server_port

        Logger.milestone_print('Running on IP: '+ip)
        Logger.milestone_print('Running on port: '+str(port))

        self.s.bind((ip,port))
        self.s.listen(100)

        while 1:
            try:
                c, addr = self.s.accept()
            except KeyboardInterrupt as e:
                print("\nctrl+c,Exiting gracefully")
                self.s.shutdown(socket.SHUT_RDWR)
                self.s.close()
                exit(0)
            # print(c)

            threading.Thread(target=self.handle_client,args=(c,addr,)).start()

    def handle_client(self,c,addr):
        # global request_count
        # print(addr)
        print(' [%d]' % (self.request_count), end ="\r") 
        self.request_count = self.request_count + 1
        Logger.debug_print("handle_client:Entry")
        received_data = c.recv(1024).decode()
        Logger.debug_print("handle_client:received_data="+received_data)
        obj = json.loads(received_data)
        Logger.debug_print(obj)
        request = obj['request']
        if(request == 'load_model_request'):
            model_type = obj['model']
            model_path = None
            if('model_path' in obj.keys()):
                model_path = obj['model_path']
            response = ''
            if request in self.callbacks :
                callback = self.callbacks[request]
                response = callback(model_type,model_path)

            Logger.debug_print("handle_client:sending pred_caption" + response)
            c.send(response.encode())
            # candidate = pred_caption.split()
            Logger.debug_print ('response:' + response)
        else:
            tensor_shape = obj['data_shape']
            zlib_compression = False
            if 'zlib_compression' in obj.keys():
                if(obj['zlib_compression'] == 'yes'):
                    zlib_compression = True

            quantized = False
            if 'quantized' in obj.keys():
                if(obj['quantized'] == 'yes'):
                    quantized = True

            reshape_image_size = obj['reshape_image_size']
            Logger.debug_print("handle_client:sending OK")
            c.send("OK".encode())

            max_data_to_be_received = obj['data_size']
            total_data = 0
            msg = bytearray()
            while 1:
                # print("handle_client:calling recv total_data=%d data_size=%d" % (total_data, max_data_to_be_received))
                if(total_data >= max_data_to_be_received):
                    Logger.debug_print("handle_client:received all data")
                    break
                data = c.recv(1024)
                # print(type(data))
                msg.extend(data)
                if not data:
                    Logger.debug_print("handle_client:while break")
                    break
                total_data += len(data)
            
            Logger.debug_print('total size of msg=%d' % (len(msg)))
            
            response = ''
            if request in self.callbacks :
                callback = self.callbacks[request]
                response = callback(msg,tensor_shape,reshape_image_size,quantized,zlib_compression)

            Logger.debug_print("handle_client:sending pred_caption" + response)
            c.send(response.encode())
            # candidate = pred_caption.split()
            Logger.debug_print ('response:' + response)