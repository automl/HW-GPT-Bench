

from gpt.utils import *
import pickle
from pl_gpt.pl_module.lm_evaluator_configurable import LanguageModelEvaluator
from pl_gpt.data.lm_datamodule_nas import PlArrowFileModule
from pl_gpt.utils.instantiate import instantiate
import pytorch_lightning as pl
import os
import logging 

class GPTProfilerPPL():
    """
    class for model profiling
    """

    def __init__(
            self,
            args,
            cfg,
            num_archs_to_evaluate = 100,
            resume_path = "none",
    ):
        super().__init__()
        # build choices dict
        self.args = args
        self.model_scale = args.model_scale
        self.choices_dict = {}
        cfg_model = cfg.model
        #self.gpu_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg_model.gpu_dtype]
        self.choices_dict['n_layer_choices'] = cfg_model.layer_choices
        self.choices_dict['n_head_choices'] = cfg_model.head_choices
        self.choices_dict['embed_dim_choices'] = cfg_model.embed_choices
        self.choices_dict['mlp_ratio_choices'] = cfg_model.mlp_ratio_choices
        self.choices_dict['bias_choices'] = cfg_model.bias_choices
        cfg.trainer.precision = "16"
        cfg.model.precision = "16"
        logger = logging.getLogger(__name__)
        cfg_lm_data = {**cfg.lm_data}
        cfg_lm_data["num_gpu_worker"] = cfg.trainer.devices * cfg.trainer.num_nodes
        self.data_module = PlArrowFileModule(**cfg_lm_data)
        assert (cfg.lm_data.max_sample_len) % 128 == 0
        print(self.data_module.seq_vocab_size)
        cfg.model.vocab_size = self.data_module.seq_vocab_size
        cfg.model.padded_vocab_size = self.data_module.trg_vocab_size
        cfg.model.max_len = cfg.lm_data.max_sample_len
        logger.info(f"#### vocab size: {self.data_module.seq_vocab_size}")
        print(cfg_model)
        self.trainer_pl  = LanguageModelEvaluator(cfg_model=cfg_model, cfg_train=cfg.train, checkpoint_path=cfg.model.checkpoint_path, py_logger=logger,val_sets_name=self.data_module.val_sets_name, ignore_index=self.data_module.ignore_index)
        self.num_archs_to_evaluate = num_archs_to_evaluate
        self.cfg_model = cfg_model
        self.ppl_bench = []
        self.archs_evaluated = []
        self.data_module.prepare_data()
        self.data_module.setup("test")

        if cfg.trainer.devices == 1:
            strategy = "ddp"

            strategy = pl.strategies.DDPStrategy(
                find_unused_parameters=True,
                static_graph=False
            )
        # strategy = pl.strategies.DeepSpeedStrategy(
        #     **cfg.deepspeed,
        #     remote_device=None,  # Initialize directly on GPUs instead of CPU (ZeRO-3)
        # )

        else:
            strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=True,
            static_graph=False
        )

        # strategy = pl.strategies.DeepSpeedStrategy(
        #     **cfg.deepspeed,
        #     remote_device=None,  # Initialize directly on GPUs instead of CPU (ZeRO-3)
        # )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        training_logger = pl.loggers.tensorboard.TensorBoardLogger(
                        save_dir=".",
                        name="",
                        version="tb",
                        #    log_graph: False
                        #    default_hp_metric: True
                        prefix="",
                        )
        self.trainer = instantiate(cfg.trainer, instance=pl.Trainer ,
        callbacks=[],
        plugins=[],
        strategy=strategy,
        logger=training_logger,)
        if resume_path!="none" and os.path.exists(resume_path):
            import pickle
            with open(resume_path,"rb") as f:
                self.ppl_bench = pickle.load(f)
            self.evaluated_archs()
            self.num_archs_to_evaluate  = self.num_archs_to_evaluate - len(self.archs_evaluated)
        else:
            self.archs_evaluated = []
        save_path = "arch_ppl_bench_" + self.model_scale
        os.makedirs(save_path,exist_ok=True)

    def evaluated_archs(self):
        self.archs_evaluated = []
        for arch in self.ppl_bench:
            self.archs_evaluated.append(arch["arch"])

    def sample_n_random_archs(self):
        self.archs_sampled = []
        i = 0
        while len(self.archs_sampled)<self.num_archs_to_evaluate:
            seed = random.randint(0,1000000)
            arch_sampled = sample_config(self.choices_dict, layer_sampling_scheme="normal", seed=seed)
            if (arch_sampled not in self.archs_sampled) and arch_sampled not in self.archs_evaluated:
                self.archs_sampled.append(arch_sampled)
                #print(len(self.archs_sampled))
                i+=1
        # save archs to pickle file
        save_path = "sampled_archs_" + self.model_scale + ".pkl"
        with open(save_path,"wb") as f:
            pickle.dump(self.archs_sampled,f)

    def reset_config(self, arch_config):
        self.cfg_model.n_embd = arch_config["sample_embed_dim"]
        self.cfg_model.n_layer = arch_config["sample_n_layer"]
        self.cfg_model.n_head = arch_config['sample_n_head']
        self.cfg_model.mlp_ratio = arch_config["sample_mlp_ratio"]
        self.cfg_model.bias = arch_config["sample_bias"]



    def create_model(self, arch_config):
        arch_config["sample_intermediate_size"] = [int(arch_config["sample_mlp_ratio"][i]*arch_config["sample_embed_dim"]) for i in range(arch_config["sample_mlp_ratio"])]
        self.model.set_sample_config(arch_config["sample_embed_dim"], arch_config["sample_intermediate_size"], arch_config["sample_num_heads"], arch_config["sample_n_layer"], arch_config["sample_bias_flag"], arch_config["sample_layer_indices"])

    def compute_metrics(self, arch_config):
        self.trainer_pl.set_sample_config(arch_config)
        self.trainer.validate(self.trainer_pl, self.data_module.val_nas_dataloader())
        # print agregated stats across all processes only once
        if os.environ.get("LOCAL_RANK") is None or os.environ.get("LOCAL_RANK") == 0:
           is_rank_zero = True
           rank = 0
        else:
           is_rank_zero = False
           rank = os.environ.get("LOCAL_RANK")
        #if is_rank_zero:
        print(self.trainer_pl.current_metrics)
        metrics_dict = {}
        metrics_dict["accuracy"] = self.trainer_pl.current_metrics["acc"]
        metrics_dict["perplexity"] = self.trainer_pl.current_metrics["ppl"]
        metrics_dict["arch"] = arch_config
        if metrics_dict not in self.ppl_bench:
           self.ppl_bench.append(metrics_dict)
           save_dir = "arch_ppl_bench_" + self.model_scale 
           print(self.ppl_bench)
           os.makedirs(save_dir,exist_ok=True)
           save_path = save_dir+"/ppl_observations_tracker_"+str(self.args.start_index)+"_"+str(self.args.end_index)+".pkl"
           #save_path = self.save_path+"/efficiency_energy_observations_tracker_"+str(self.args.start_index)+"_"+str(self.args.end_index)+".pkl"
           #print(self.lat_bench)
           with open(save_path,"wb") as f:
            pickle.dump(self.ppl_bench,f)

    def run(self):
        path_pickle = "sampled_archs_" + self.model_scale + ".pkl"
        if os.path.exists(path_pickle):
            with open(path_pickle,"rb") as f:
                self.archs_sampled = pickle.load(f)
            self.archs_sampled = self.archs_sampled[self.args.start_index:self.args.end_index]
            if self.archs_evaluated!=[]:
                self.archs_sampled = [arch for arch in self.archs_sampled if arch not in self.archs_evaluated]

        else:
            self.sample_n_random_archs()

        for arch_config in self.archs_sampled:
            self.compute_metrics(arch_config)
            #break

if __name__=='__main__':
   from pl_gpt.utils.configuration import Config
   import argparse
   parser = argparse.ArgumentParser(description='GPT Profiler')
   parser.add_argument('--config', type=str, default="config/juwels_owt_sw_s_eval.yaml", help='path to config file')
   parser.add_argument('--start_index', type=int, default=0, help='start index')
   parser.add_argument('--end_index', type=int, default=10000, help='end index')
   parser.add_argument('--model_scale', type=str, default="s", help='model scale')
   args = parser.parse_args()
   config = Config(config_file=args.config)

   config_model = config.model
   config_train = config.train
   save_dir =  "arch_ppl_bench_" + args.model_scale
   os.makedirs(save_dir,exist_ok=True) 
   args.resume = save_dir + "/ppl_observations_tracker_"+str(args.start_index)+"_"+str(args.end_index)+".pkl"
   #print(config_model)
   profiler = GPTProfilerPPL(args, config, resume_path=args.resume)
   profiler.run()