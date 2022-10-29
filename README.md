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
			--mount type=bind,source=/host_path/to/data,target=/container_path/to/data \
			lgl603/deepetpicker:latest
	```

- The option `--shm-size` is used to set the required size of shared momory of the docker containers.
- The option `--mount` is used to mount a file or directory on the host machine into the Docker container, where `source` denotes host machine and  `target` denotes the container.

5. The DeepETPicker can be used directly in this machine, and it also can be used by a machine in the same LAN.
   - Directly open DeepETPicker in this machine: 
   ```bash
   ssh -X test@172.17.0.2 DeepETPicker
   ```
   - Connect to this server remotely and open DeepETPicker software via a client machine:
   ```bash
   ssh -p 50022 test@ip DeepETPicker
   ```
	Here `ip` is the IP address of the server machine，password is `password`.


## **Particle picking tutorial**
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
	│   │   └── tomo2.mrc
	│   │   └── tomo3.mrc
	│   │   └── ...
	```

	For above data, tomo1.mrc can be used as train/val dataset. tomo2.mrc, tomo3.mrc and tomo4.mrc are test dataset, as they have no matual annotation.
	
	<br>

	For the sample dataset of EMPAIR-10045, the data structure is as follows:
	```
	├── /base/path
	│   ├── raw_data
	│   │   ├── IS002_291013_005_iconmask2_norm_rot_cutZ.coords
	│   │   └── IS002_291013_005_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_006_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_007_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_008_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_009_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_010_iconmask2_norm_rot_cutZ.mrc
	│   │   └── IS002_291013_011_iconmask2_norm_rot_cutZ.mrc
	``` 			

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
	
<div align=center>
	<img src="https://github.com/cbmi-group/DeepETPicker/blob/main/images/Training.gif" width="80%"> 
</div>

In practice, default parameters can give you a good enough result.

*Training parameter description:*

- Dataset name: e.g. SHREC_2021_train
- Train dataset ids: datasets used for training. You can click `Dataset list` to obain the dataset ids firstly.
- Val dataset ids: datasets used for validation. You can click `Dataset list` to obain the dataset ids firstly.
- Number of classes:  particle classes you want to pick
- Batch size: a number of samples processed before the model is updated. It is determined by your GPU memory, reducing this parameter might be helpful if you encounter out of memory error.
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

If you use this code for your research, please cite our paper [DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning]().

```
@article{DeepETPicker,
  title={DeepETPicker: Fast and accurate 3D particle picking for cryo-electron tomography using weakly supervised deep learning},
  author={Guole Liu, Tongxin Niu, Mengxuan Qiu, Fei Sun, and Ge Yang},
  journal={bioaxiv},
  year={2022}
}
```
