import sys
sys.path.append("..")
from configs.c2l_10045_New_bin8_mask3 import pre_config
from utils.coords2labels import Coord_to_Label
from utils.coord_gen import coords_gen
from utils.normalization import InputNorm

# 初始化坐标文件为整数
coords_gen(pre_config["coord_path"],
           pre_config["base_path"])

# 归一化
pre_norm = InputNorm(pre_config["tomo_path"],
           pre_config["tomo_format"],
           pre_config["base_path"],
           pre_config["norm_type"])
pre_norm.handle_parallel()

# 根据coords产生labels
c2l = Coord_to_Label(base_path=pre_config["base_path"],
                     coord_path=pre_config["coord_path"],
                     coord_format=pre_config["coord_format"],
                     tomo_path=pre_config["tomo_path"],
                     tomo_format=pre_config["tomo_format"],
                     num_cls=pre_config["num_cls"],
                     label_type=pre_config["label_type"],
                     label_diameter=pre_config["label_diameter"],
                     )
c2l.gen_labels()

# 根据coords产生ocps
if pre_config["label_diameter"] !=pre_config["ocp_diameter"]:
    c2l = Coord_to_Label(base_path=pre_config["base_path"],
                         coord_path=pre_config["coord_path"],
                         coord_format=pre_config["coord_format"],
                         tomo_path=pre_config["tomo_path"],
                         tomo_format=pre_config["tomo_format"],
                         num_cls=pre_config["num_cls"],
                         label_type=pre_config["ocp_type"],
                         label_diameter=pre_config["ocp_diameter"],
                         )
    c2l.gen_labels()