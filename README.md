# **DeepETpicker**

A deep learning based open-source software with a friendly user interface to pick 3D particles rapidly and accurately from cryo-electron tomograms. With the advantages of weak labels, lightweight architecture and GPU-accelerated pooling operations, the cost of annotations and the time of computational inference are significantly reduced while the accuracy is greatly improved by applying a Gaussian-type mask and using a customized architecture design.

[DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning]()

Authors: Guole Liu, Tongxin Niu, Mengxuan Qiu, Yun Zhu, Fei Sun, and Ge Yang

**Note**: DeepETPicker is a Pytorch implementation. 

## **Setup**

### **Prerequisites**

- Linux
- NVIDIA GPU

### **Installation**

The following steps are required in order to run DeepETPicker:
1. Install [Docker](https://www.docker.com/)

	Note: docker engine version shuold be >= 19.03. The size of Docker mirror of Deepetpicker is 7.21 GB, please ensure that there is enough memory space.

2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

3. Download Docker image of DeepETPicker.

	```bash
	docker pull docker.io/lgl603/deepetpicker:latest
	```

4. Run the Docker image of DeepETPicker.

	```bash
	docker run --gpus=all -itd \
			--restart=always \
			--shm-size=100G \
			-e DISPLAY=unix$DISPLAY \
			--name deepetpicker \
			-p 50022:22 \
			--mount type=bind,source='/host_path/to/data',target='/container_path/to/data' \
			lgl603/deepetpicker:latest
	```

- The option `--shm-size` is used to set the required size of shared momory of the docker containers.
- The option `--mount` is used to mount a file or directory on the host machine into the Docker container, where `source='/host_path/to/data'` denotes denotes the data directory really existed in host machine. `target='/container_path/to/data'` is the data directory where the directory `'/host_path/to/data'` is mounted in the container. 

	**Note: `'/host_path/to/data'` should be replaced by the data directory real existed in host machine. For convenience, `'/container_path/to/data'` can set the same as `'/host_path/to/data'`**
	<img width="960" alt="image" src="https://user-images.githubusercontent.com/16335327/198990001-b04fbd1e-c284-482a-81f9-c266fc957a42.png">


5. The DeepETPicker can be used directly in this machine, and it also can be used by a machine in the same LAN.
   - Directly open DeepETPicker in this machine: 
   ```bash
   ssh -X test@'ip_address' DeepETPicker
   # where the 'ip_address' of DeepETPicker container can be obtained as follows:
   docker inspect --format='{{.NetworkSettings.IPAddress}}' deepetpicker
   ```
   <img width="702" alt="image" src="https://user-images.githubusercontent.com/16335327/198967756-3a409b6f-bc19-42cd-83ec-d4cb67776b58.png">

   
   - Connect to this server remotely and open DeepETPicker software via a client machine:
   ```bash
   ssh -X -p 50022 test@ip DeepETPicker
   ```
	Here `ip` is the IP address of the server machine，password is `password`.

`Installation time`: the size of Docker mirror of Deepetpicker is 7.21 GB, and the installation time depends on your network speed. When the network speed is fast enough, it can be configured within a few minutes.

## **Particle picking tutorial**

Detailed tutorials for two sample datasets of [SHREC2021](https://github.com/cbmi-group/DeepETPicker/blob/main/tutorials/A_tutorial_of_particlePicking_on_SHREC2021_dataset.md) and [EMPIAR-10045](https://github.com/cbmi-group/DeepETPicker/blob/main/tutorials/A_tutorial_of_particlePicking_on_EMPIAR10045_dataset.md) are provided. Main steps of DeepETPicker includeds preprocessing, traning of DeepETPicker, inference of DeepETPicker, and particle visualization.

### **Preprocessing**
- Data preparation
  
	Before launching the graphical user interface, we recommend creating a single folder to save inputs and outputs of DeepETpicker. Inside this base folder you should make a subfolder to store raw data. This raw_data folder should contain: 
  - tomograms(with extension .mrc or .rec)
  - coordinates file with the same name as tomograms except for extension. (with extension *.csv, *.coords or *.txt. Generally, *.coords is recoommand.).  

	<br>

  Here, we provides two sample datasets of EMPIAR-10045 and SHREC_2021 for particle picking to enable you to learn the processing flow of DeepETPicker better and faster. The sample dataset can be download in one of two ways:
  - Baidu Netdisk Link: [https://pan.baidu.com/s/1aijM4IgGSRMwBvBk5XbBmw](https://pan.baidu.com/s/1aijM4IgGSRMwBvBk5XbBmw ); verification code: cbmi
  - Microsoft onedrive Link: [https://1drv.ms/u/s!AmcdnIXL3Vo4hWf05lhsQWZWWSV3?e=dCWvew](https://1drv.ms/u/s!AmcdnIXL3Vo4hWf05lhsQWZWWSV3?e=dCWvew); verification code: cbmi

- Data structure
  
  The data should be organized as follows:
	```
	├── /base/path
	│   ├── raw_data
	│   │   ├── tomo1.coords
	│   │   └── tomo1.mrc
	│   │   └── tomo2.coords
	│   │   └── tomo2.mrc
	│   │   └── tomo3.mrc
	│   │   └── tomo4.mrc
	│   │   └── ...
	```

	For above data, `tomo1.mrc` and `tomo2.mrc` can be used as train/val dataset, since they all have coordinate files (matual annotation). If a tomogram has no matual annotation (such as `tomo3.mrc`), it only can be used as test dataset.
	
	<br>


- Input & Output

	<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/Preprocessing.gif" width="80%"> 
	</div>

	Launch the graphical user interface of DeepETPicker. On the `Preprocessing` page, please set some key parameters as follows:
	- `input`
		- Dataset name: e.g. SHREC_2021_preprocess
		- Base path: path to base folder
		- Coords path:  path to raw_data folder 
		- Coords format: .csv, .coords or .txt
		- Tomogram path: path to raw_data folder
		- Tomogram format: .mrc or .rec
		- Number of classes:  multiple classes of macromolecules also can be localized separately
	- `Output`
		- Label diameter(in voxels):  smaller than the average diameter of the targets
		- Ocp diameter(in voxels): this value can be the average diameter of the targets. For particles of multi-classes, their diameters should be separated with a comma.
		- Configs: if you click 'Save configs', it would be the path to the file which contains all the parameters filled in this page  


### **Training of DeepETPicker**
	
<br>

	Note: Before `Training of DeepETPicker`, please do `Preprocessing` first to ensure that the basic parameters required for training are provided.

<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/Training.gif" width="80%"> 
</div>

In practice, default parameters can give you a good enough result.

*Training parameter description:*

- Dataset name: e.g. SHREC_2021_train
- Dataset list: get the list of train/val tomograms. The first column denotes particle number, the second column denotes tomogram name, the third column denotes tomogram ids. If you have n tomograms, the ids will be {0, 1, 2, ..., n-1}.
- Train dataset ids: tomograms used for training. You can click `Dataset list` to obain the dataset ids firstly. One or multiple tomograms can be used as training tomograms. But make sure that the `traning dataset ids` are selected from {0, 1, 2, ..., n-1}, where n is the total number of tomograms obtained from `Dataset list`. Here, we provides two ways to set dataset ids:
  - 0, 2, ...: different tomogram ids are separated with a comma.
  - 0-m: where the ids of {0, 1, 2, ..., m-1} will be selected. Note: this way only can be used for tomograms with continuous ids.
- Val dataset ids: tomograms used for validation. You can click `Dataset list` to obain the dataset ids firstly. Note: only one tomogram can be selected as val dataset.
- Number of classes:  particle classes you want to pick
- Batch size: a number of samples processed before the model is updated. It is determined by your GPU memory, reducing this parameter might be helpful if you encounter out of memory error.
- Patch size: the sizes of subtomogram. It needs to be a multiple of 8. It is recommended that this value is not less than 64, and the default value is 72.
- Padding size: a hyperparameter of overlap-tile strategy. Usually, it can be from 6 to 12, and the default value is 12.
- Learning rate:  the step size at each iteration while moving toward a minimum of a loss function.
- Max epoch: total training epochs. The default value 60 is usually sufficient.
- GPU id: the GPUs used for training, e.g. 0,1,2,3 denotes using GPUs of 0-4. You can run the following command to get the information of available GPUs: nvidia-smi.
- Save Configs: save the configs listed above. The saved configurations contains all the parameters filled in this page, which can be directly loaded via *`Load configs`* next time instead of filling them again.

### **Inference of DeepETPicker**
	
<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/Inference.gif" width="80%"> 
</div>
	
In practice, default parameters can give you a good enough result.

*Inference parameter description:*

- Train Configs: path to the configuration file which has been saved in the `Training` step
- Networks weights: path to the model which has be generated in the `Training` step
- Patch size & Pad_size: tomogram is scanned with a specific stride S and a patch size of N in this stage, where S = N - 2Pad_size.
- GPU id: the GPUs used for inference, e.g. 0,1,2,3 denotes using GPUs of 0-4. You can run the following command to get the information of available GPUs: nvidia-smi.

### **Particle visualization and mantual picking**

<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/Visualization.gif" width="90%"> 
</div>

- *Showing Tomogram*

	You can click the `Load tomogram` button on this page to load the tomogram.

- *Showing Labels*
	
	After loading the coordinates file by clicking `Load labels`, you can click `Show result` to visualize the label. The label's diameter and width also can be tuned on the GUI.

- *Parameter Adjustment*

	In order to increase the visualization of particles, Gaussian filtering and histogram equalization are provided:
	- Filter: when choosing Gaussian, a Gaussian filter can be applied to the displayed tomogram, kernel_size and sigma(standard deviation) can be tuned to adjust the visual effects
	- Contrast: when choosing hist-equ, histogram equalization can be performed 
	

- *Mantual picking*
	
	After loading the tomogram and pressing ‘enable’, you can pick particles manually by double-click the left mouse button on the slices. If you want to delete an error labeling, just right-click the mouse.	You can specify a different category id per class. Always remember to save the resuls when you finish.

- *Position Slider*
	
	You can scan through the volume in x, y and z directions by changing their values. For z-axis scanning, shortcut keys of Up/Down arrow can be used.
	
# Citation

If you use this code for your research, please cite our paper [DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning](https://github.com/cbmi-group/DeepETPicker).

```
@article{DeepETPicker,
  title={DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning},
  author={Guole Liu, Tongxin Niu, Mengxuan Qiu, Yun Zhu, Fei Sun, and Ge Yang},
  journal={bioaxiv},
  year={2022}
}
```
