Explaining the structure of the directory:
1.dataset_hate: Directory which consists of the dataset for the object detection and caption hinderance task.

2.datasets/coco128: Directory which consists of coco128 images dataset.
3.img: Directory which consists of the dataset for the hateful-meme classification tasks. Dataset fetched from Facebook meme challenge.

4.meme_classifier: Directory which consists of the dataset for the meme-classification task along with the trained model.h5 (Meme or not)

5.output_images: Directory which consists of output images for various tasks

6.plots: Directory which consists of plots obtained from various tasks

7.results: Directory which consists of the results obtained from object detection task.

8.runs: Directory which pertains to the intialization of YOLOV8 model for object detection containing config files and weights obtained after training.
This directory also contains plots obtained while training the YOLOV8 model.

9.Various text files pertain to the task of hateful meme classification.

10.comparison.ipynb: Python notebook for comparison analysis of the meme-based model and text-based model for hateful-meme classification.

11.hateful_meme_classifier.ipynb: Python notebook for the meme-based model classification of hateful memes.

12.meme_classifier.ipynb: Python notebook for the meme classifier(Meme or not)

13.model.h5-trained model for the meme-based model classification of hateful memes.

14.obj_detection.ipynb/py: Script for object detection task.
Note that while working with the object detection task certain parts of the code must be commented out and uncommented out at certain points.

15.Paper_Reading_Task.pdf: PDF of analysis of paper reading task.

16.Precog_Task_Report: Report of Precog Task

17.test_seen.jsonl: jsonl file required for intialising the test dataset for the hateful-meme classification task.

18.text_caption_classifier.ipynb: Python notebook for text-based classifier of hateful-meme classification task.

19.text_extractor.ipynb: Python notebook for extracting text from images.

20.train.jsonl: jsonl file required for initialising the training dataset for the hateful-meme classification task.

21.ASSUMPTION:Some of the directories such as dataset_hate, datasets/coco128, img and meme_classifier are too big to be uploaded into github and will be taken directly from local storage during evaluation.

