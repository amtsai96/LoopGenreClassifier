# Loops Genre Classifier
Music Loop Genre Classification with CNN

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

## Environment
- Python 3.7
- Tensorflow 1.14.0
- Keras 2.2.5
- Librosa 0.6.3
- Numpy 1.16.4
- Pandas 0.24.2
- Seaborn 0.9.0
- Matplotlib 3.1.0

## Dataset
- Loop Music on Looperman.com (grouped by genres).
- Data information in ```looperman_dataset_info``` folder.
- You can plot your data information with ```plot_data_info.py```.
---

#### You may need to prepare ...
- ```account.xlsx``` a file to list your account and passwords (to log in semi-automatically)
- ```chromedriver```

### Loop Data Crawler
- To Crawl loops, run ```python loop_crawler.py```. (Remember to change the data path in the file)
- ```get_url_of_first_page.py``` can get the urls on the first page of a specific genre.
---
### Loop Data Genre Classifier
- ```gen_dataset.py``` is for generating Mel-Spectrogram and dataset text files.
- Run the code ```python CNN_train.py``` to train the model.


[contributors-shield]: https://img.shields.io/github/contributors/amtsai96/LoopGenreClassifier.svg?style=flat-square
[contributors-url]: https://github.com/amtsai96/LoopGenreClassifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/amtsai96/LoopGenreClassifier.svg?style=flat-square
[forks-url]: https://github.com/amtsai96/LoopGenreClassifier/network/members
[stars-shield]: https://img.shields.io/github/stars/amtsai96/LoopGenreClassifier.svg?style=flat-square
[stars-url]: https://github.com/amtsai96/LoopGenreClassifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/amtsai96/LoopGenreClassifier.svg?style=flat-square
[issues-url]: https://github.com/amtsai96/LoopGenreClassifier/issues
