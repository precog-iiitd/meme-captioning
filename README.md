Meme Face Emotion Captioning
===================
*Generate rich emotion captions for a given meme face image*

#### Paper
This is the code repository for the paper: *Towards increased accessibility of Meme Images with the help of Rich Face Emotion Captions*, **ACM Multimedia 2019**.

Prerequisites
-------------
- Python 2.7 (the only version supported by coco-caption)
- Install the coco-caption evaluation tool into `convcap/third_party` folder.
- Install necessary packages using `pip install -r requirements.txt`.
- Download the [`Glove6B-300d word vectors`](http://nlp.stanford.edu/data/glove.6B.zip) into the `convcap/data` folder.

Getting the weights
----------
Download the weights of the trained face emotion captioning models into the `convcap/saved_models` folder
| `Model` | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROGUE-L | METEOR | CIDEr | SPICE | `Link` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ranking |  **0.48**  |  **0.28**  |  **0.17**  |  **0.12**  |  **0.36**  |  **0.15**  |  **0.28**  |  **0.14**  | [`Drive`](https://drive.google.com/open?id=1sLooGw56h-N5chZKDVRAnWLL4fYI77JP)
| Regression  | 0.44 | 0.24 | 0.14 | 0.09 | 0.33 | 0.13 | 0.18 | 0.12 | [`Drive`](https://drive.google.com/file/d/1xqiRvZiyj672TQrEZ1-fBqnJThcOx2Nv/view?usp=sharing)

Generating face emotion captions using pretrained models
-------
Our model takes a face crop image as input and generates a caption for the emotion in the face. Please refer to face crop samples in `convcap/data/manual_test/` folder. You can use `dlib` to automatically crop faces if necessary. 
```bash
cd convcap
python captioner.py --img <path_to_face_crop> --checkpoint <path_to_checkpoint>

#### know about more options
python captioner.py -h
```

Two-stage Training pipeline
-------
As described in the paper, **Stage 1** involves training a CNN on AffectNet to learn face emotion features. The code to pretrain a CNN on AffectNet is provided in the `affectnet_pretraining/` folder. If you would like to skip this and use our pretrained CNNs, just download the pretrained CNNs given below and go to **Stage 2**. 

#### Pre-training on AffectNet
If you would like to skip this step and use our pretrained CNN instead, download the desired model below and place them in `convcap/saved_models`

| `Model` | `Link` |
| :-------: | :------: |
| **Ranking CNN** | [`Drive`](https://drive.google.com/open?id=1KTFpeq27m1XXW2sWzdKdwzwq3zck__cm)
| Regression CNN | [`Drive`](https://drive.google.com/open?id=1TFO8-INXk_mnENMjwRs5mPlsRxZjuBeB)

```bash
# Please download the AffectNet dataset into the affectnet_pretraining/data folder. And then extract and store faces:
python main.py --mode preprocess --data_root data/ --img_dir <folder_name containing images> 

# Faces will be cropped, resized and saved in <data_root>/faces. Run training:
python main.py --mode train --data_root data/ --img_dir faces/ --config <regression | ranking>

### More options for the above steps
python main.py --help
```
#### Training the Emotion Captioner
To train the emotion captioner, please first request and download the data from the [`project page`](). Extract the contents to `dataset/` folder. 

```bash
# First create the train, test and holdout splits:
python create_splits.py

# Start the training:
python main.py --mode train --imgcnn <path_to_pretrained CNN from Step 1> 

### More options for training
python main.py --help
```