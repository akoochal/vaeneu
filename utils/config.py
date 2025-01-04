import yaml
import numpy as np
import pprint

class Config:
    def __init__(self,config_path,device) -> None:
        with open(config_path,"rb") as f:
            self.raw_config = yaml.safe_load(f)
        self.model_type = self.raw_config["type"]
        self.model = self._ModelConfig(self.raw_config["MODEL"],self.model_type,device)
        self.train = self._TrainConfig(self.raw_config["TRAIN"],self.model_type)

    class _ModelConfig:
        def __init__(self,conf_dict,model_type,device) -> None:
            self.device = device
            self.history_window_size = conf_dict.get("history_window_size",None)
            self.horizon = conf_dict.get("horizon",None)
            self.optimizer = conf_dict["optimizer"]
            self.lr = conf_dict["lr"]

            if model_type == "forgan":
                self.gen = self.get_module(conf_dict["gen"])
                self.dsc = self.get_module(conf_dict["dsc"])
                self.loss = conf_dict["loss"]
                if conf_dict["loss"] == "wgp":
                    self.lambda_gp = conf_dict["lambda_gp"]
                
            elif model_type == "vaeneu":
                self.enc = self.get_module(conf_dict["enc"])
                self.dec = self.get_module(conf_dict["dec"])
                self.repeat_factor = conf_dict["repeat_factor"]

        def get_module(self,conf_dict):
            if conf_dict["type"] in ["LSTM","GRU"]:
                return self._RNN_Config(conf_dict)
            elif conf_dict["type"] == "TCN":
                return self._TCN_Config(conf_dict)

        class _RNN_Config:
            def __init__(self,conf_dict) -> None:
                self.type = conf_dict["type"]
                self.hidden_size = conf_dict["hidden_size"]
                self.num_layers = conf_dict["num_layers"]
                if conf_dict.get("noise_size",None) is not None:
                    self.noise_size = conf_dict["noise_size"]

        class _TCN_Config:
            def __init__(self,conf_dict) -> None:
                self.type = conf_dict["type"]
                self.hidden_size = conf_dict["hidden_size"]
                self.num_layers = conf_dict["num_layers"]
                self.kernel_size = conf_dict["kernel_size"]
                if conf_dict.get("noise_size",None) is not None:
                    self.noise_size = conf_dict["noise_size"]
    
    class _TrainConfig:
        def __init__(self,conf_dict,model_type):
            self.n_steps= conf_dict.get("n_steps",1000)
            self.val_interval= conf_dict.get("val_interval",10)
            self.n_samples= conf_dict.get("n_samples",100)
            self.skip_val= conf_dict.get("skip_val",0)
            self.early_stopping_tolerance= conf_dict.get("early_stopping_tolerance",np.inf)
            self.batch_size = conf_dict.get("batch_size",128)
            self.model_path= conf_dict.get("model_path","")
            self.log_path= conf_dict.get("log_path","")
            self.screen_log=conf_dict.get("screen_log",False)
            self.file_log=conf_dict.get("file_log",False)
            if model_type == "forgan":
                self.d_iter= conf_dict.get("d_iter",1)

    def get_dict(self) -> dict:
        model_dict = {}
        for key, value in vars(self.model).items():
            if key != "device":
                if hasattr(value, '__dict__'):
                    model_dict[key] = value.__dict__
                else:
                    model_dict[key] = value
        return {"type":self.model_type,
                    "MODEL":model_dict,
                    "TRAIN":vars(self.train)}

    def __str__(self) -> str:
        return pprint.pformat(self.get_dict(),sort_dicts=False,indent=4)
    
    def save_config(self,config_path):
        with open(config_path,"w") as f:
            yaml.dump(self.get_dict(),f)

    def auto_config(self):
        if self.model.history_window_size == 0 or self.model.history_window_size is None:
            raise Exception("The history window size should be set for auto configuration")
        
        if self.model.horizon == 0 or self.model.horizon is None:
            raise Exception("The horizon should be set for auto configuration")
        
        if self.model_type == "forgan":
            if self.model.gen.type in ["LSTM","GRU"]:
                self.model.gen.hidden_size = self.model.history_window_size // 2
                self.model.gen.noise_size = self.model.history_window_size // 2
            elif self.model.gen.type == "TCN":
                self.model.gen.hidden_size = 2*int(np.ceil(np.log2(self.model.history_window_size)))
                self.model.gen.noise_size = 2*int(np.ceil(np.log2(self.model.history_window_size)))

                g_n_layers,g_kernel_size = get_n_k(self.model.history_window_size)
                self.model.gen.num_layers = g_n_layers
                self.model.gen.kernel_size = g_kernel_size
            else:
                raise Exception("Generator type {} is not implemented".format(self.model.gen.type))
            
            if self.model.dsc.type in ["LSTM","GRU"]:
                self.model.dsc.hidden_size = (self.model.history_window_size + self.model.horizon)

            elif self.model.dsc.type == "TCN":
                self.model.dsc.hidden_size = 4*int(np.ceil(np.log2((self.model.history_window_size + self.model.horizon))))

                d_n_layers,d_kernel_size = get_n_k((self.model.history_window_size + self.model.horizon))
                self.model.dsc.num_layers = d_n_layers
                self.model.dsc.kernel_size = d_kernel_size
            else:
                raise Exception("Discriminator type {} is not implemented".format(self.model.dsc.type))
            
        elif self.model_type == "vaeneu":
            if self.model.dec.type in ["LSTM","GRU"]:
                self.model.dec.hidden_size = self.model.history_window_size // 2
                self.model.dec.noise_size = self.model.history_window_size // 2
            elif self.model.dec.type == "TCN":
                self.model.dec.hidden_size = 2*int(np.ceil(np.log2(self.model.history_window_size)))
                self.model.dec.noise_size = 2*int(np.ceil(np.log2(self.model.history_window_size)))

                dec_n_layers,dec_kernel_size = get_n_k(self.model.history_window_size)
                self.model.dec.num_layers = dec_n_layers
                self.model.dec.kernel_size = dec_kernel_size
            else:
                raise Exception("Decoder type {} is not implemented".format(self.model.dec.type))
            
            if self.model.enc.type in ["LSTM","GRU"]:
                self.model.enc.hidden_size = (self.model.history_window_size + self.model.horizon)

            elif self.model.enc.type == "TCN":
                self.model.enc.hidden_size = 4*int(np.ceil(np.log2((self.model.history_window_size + self.model.horizon))))

                enc_n_layers,enc_kernel_size = get_n_k((self.model.history_window_size + self.model.horizon))
                self.model.enc.num_layers = enc_n_layers
                self.model.enc.kernel_size = enc_kernel_size
            else:
                raise Exception("Encoder type {} is not implemented".format(self.model.enc.type))

def get_n_k(l):
    est_k=5
    est_n = np.log2(((l - 1)/(2*(est_k-1)))+1)
    n = np.ceil(est_n)
    k = 1 + ((l - 1)/(2*(2**n - 1)))
    k = np.ceil(k)
    return int(n), int(k)