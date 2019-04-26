# TensorFlow-Custom-Object-Detection
Object Detection using TensorFlow-Object-Detection_API

## **Installation & Requirements:**
<br/><br/>
1. First things first:
```
git clone https://github.com/pivotchain/POI-face-recognition/tree/Training_Codes/tensorflow_api
```
<br/><br/>
2. cd POI-face-recognition && git checkout Training_Codes && cd tensorflow_api
<br/><br/>
3. Create virtual enviroment by executing :
```
conda create -n TensorflowTrainingEnv python==3.6
```
<br/><br/>
4. Activate created enviroment:
```
source activate TensorflowTrainingEnv
```
<br/><br/>

5. Run below command in same Command prompt to install all required packages:
```
pip install -r requirement.txt
```
<br/><br/>

6.When running locally, slim directories should be appended to PYTHONPATH. This can be done by running the following command:
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
>Note: This command needs to run from every new terminal you start. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file, replacing `pwd` with the absolute path of tensorflow_api on your system._
<br/><br/>

7.You can test that you have correctly installed the Tensorflow Object Detection API by running the following command:
```
python object_detection/builders/model_builder_test.py
```

## **Setting train/test data & Creating `.csv` files**
<br/><br/>
1. Place train `.jpg` & their repective `.xml` files(labels) in train folder in images folder.
<br/><br/>
2. Place test `.jpg` & their repective `.xml` files(labels) in test folder images folder.
> **Note :**_For creating .xml files(labels) you can use [this](https://github.com/tzutalin/labelImg) great open source image labelling tool. Repositories README.md file has information on installing & using it._
<br/><br/>
4. Run the following command to create `train.csv` & `test.csv` files.These file will be created under images folder.
```
python parse_xml_to_csv.py
```
<br/><br/><br/>
## **Creating TensorFlow Records**
> **Note:**_Input for TensorFlow model_
<br/><br/>
1. Open `create_tf_record.py` file & replace the label map in `class_text_to_int()` function on `line 33`, according to classes present in your images & save the file.
<br/><br/>
2. Run the following command to create `train.record`
```
python create_tf_record.py --csv_input=./images/train_labels.csv --image_dir=./images/train --output_path=./images/train.record
```
<br/><br/>
3. Run the following command to create `test.record`
```
python create_tf_record.py --csv_input=./images/test_labels.csv --image_dir=./images/test --output_path=./images/test.record
```
<br/><br/><br/>
## **Configuring label map**
<br/><br/>
1. Open `labelmap.pbtxt` file in `traning` folder
<br/><br/>
2. Change the label map present according to classes you have in your images & save the file
<br/><br/><br/>
## **Download Base Model**
<br/><br/>
1. Go to [model zoo](https://github.com/tensorflow/models/blob/99256cf470df6af16808eb0e49a8354d2f9beae2/research/object_detection/g3doc/detection_model_zoo.md) & download `faster_rcnn_inception_v2_coco` (or any other model of your choice).
<br/><br/>
2. Extract theis folder to `./base_models` this folder
<br/><br/><br/>
## **Configuring object detection training pipeline**
<br/><br/>
1. Open `samples\configs` folder and copy `faster_rcnn_inception_v2_pets.config` (or file for any other model that you use)
<br/><br/>
2. paste this file in `training` directory
<br/><br/>
3. Open this file & edit the following:
    - num_classes : to the number of classes you want to detect
    - fine_tune_checkpoint : set path to  model.ckpt of any base model that you use.
    - input_path(train_input_reader) : in train_input_reader block set input_path to `train.record` file generated in previous step.
    - label_map_path(train_input_reader) : in train_input_reader block set path to `labelmap.pbtxt` file created in above step.
    - num_examples : set the number to number of test images you will be using
    - input_path(eval_input_reader) : in eval_input_reader block set input_path to `test.record` file generated in previous step.
    - label_map_path(eval_input_reader) : in eval_input_reader block set path to `labelmap.pbtxt` file created in above step.
    - num_examples : set the number to number of test images you will be using.`
<br/><br/><br/>
## **Training**
<br/><br/>
1. Execute following command while in `./tensorflow_api` directory
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

```
> **Note:**_Initialization takes upto 30 secs before actual training starts_
<br/><br/>
2. Wait until loss is decreasing consistently (can take upto hours, differs from model to model)
<br/><br/>
3. Once satisfied with loss value, press `control + c` to stop training
<br/><br/><br/>
## **Using Tensorboard(for graph visualization)**
<br/><br/>
1. In a new terminal navigate to `./tensorflow_api` directory
<br/><br/>
2. Execute following command:
```
tensorboard --logdir=training
```
<br/><br/>
3. This command will generates a URL, without stopping the process copy that url and paste into browser
<br/><br/>
4. Tensorboard page will open up and you can visualize a lot of thing about currently being trained model, eg. different loss graphs.
<br/><br/><br/>
## **Exporting Inference graph**
<br/><br/>
1. Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the `\object_detection` folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered `.ckpt` file in the `training` folder. You might also change `--pipeline_config_path` parameters according to the model you use.
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

```
<br/><br/>
2. This creates `frozen_inference_graph.pb` file in `inference_graph` directory
<br/><br/><br/>
