# Bbox compositions clustering
A tool for clustering bboxes compositions of different classes (by bboxes features and/or images features).  
It can be useful if you have a lot of images/other data with bboxes (for example, different fields in some files), and you need to find dependencies between them.
Kmeans, DBSCAN, OPTICS methods are supported.

## Usage
Environment: Python 3.11.5

1. Install requirements: `pip install -r requirements.txt`
2. Upload images with drawn bboxes to the dataset folder (umages must be the same size, each bbox class should be drawn in a unique color, with transparency to support the overlap)
3. Upload a dataframe with all information about the bboxes in required format (described below)
4. Specify all the required values in the `config.py`
5. Run `main.py` and follow the instructions

## Input df format

| file         | x1  | y1 | x2  | y2 |angle| depth | cls |
|--------------|-----|----|-----|----|-----|-------|-----|
| 4e060261.jpg | 889 | -72| 1321| 173| 6.25| 0     | 0   |
| cfb4c772.jpg | 880 | 473| 1125| 637| 0   | 0.6   | 0   |
| 65b269b9.jpg | 783 | 355| 1158| 566| 0   | 0.2   | 1   |
| aa04df5a.jpg | 222 | 291| 566 | 487| 0   | 0.37  | 0   |

Where columns:
* `file` - image with bboxes
* `x1, y1, x2, y2` - left/top/right/bottom bbox coordinates
* `angle` - rotation angle of bbox
* `depth` - z-order (depth) of bbox
* `cls` - bbox class

## Example images

<p align="center">
  <img src="example_dataset/4e9a9efca548b75b8dc0aad95eb0ec87.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/4e0602615b49d1e2c299d5ac70bfeef9.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/6f7f1def16906a171025b94f02e8ee59.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/7ad5e9b178299a2150dd0c6ce7421445.jpg" width="120" height="60" style="border: 1px solid black;"/>
</p>
<p align="center">
  <img src="example_dataset/7f79659297ecf2fdd0ff7705754be504.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/50b650d4eb8d8e6e3bc5e9cd5aea567c.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/65b269b9b57b8d65f5651d1e33069b47.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/34369dc2655c1755842dd735475392d7.jpg" width="120" height="60" style="border: 1px solid black;"/>
</p>
<p align="center">
  <img src="example_dataset/72995a2269132a8239910ce37db9d9cd.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/a39a499ee25547082cf02572963949a6.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/aa04df5a7f397cd0d60e04ee1fe542bf.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/b10bfbb5dab9e1fa626fbea05843e958.jpg" width="120" height="60" style="border: 1px solid black;"/>
</p>
<p align="center">
  <img src="example_dataset/ba2e2f22c502924d3b8c35a1278808ae.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/cfb4c772f03e90e0457ad5e365e15e6f.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/d25e58d3f4ac4e69674afaa141c78f74.jpg" width="120" height="60" style="border: 1px solid black;"/>
  <img src="example_dataset/f9e42d469127a83b89532c67dbc4ba81.jpg" width="120" height="60" style="border: 1px solid black;"/>
</p>

## Useful links
* [PCA](https://365datascience.com/tutorials/python-tutorials/pca-k-means/)
* [Multidimensional Data Analysis](https://www.geeksforgeeks.org/multidimensional-data-analysis-in-python/)
* [Kmeans](https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240)
* [Drawing bboxes](https://stackoverflow.com/questions/68875941/how-to-put-a-specific-coordinate-of-a-small-image-to-a-specific-coordinate-of-a)