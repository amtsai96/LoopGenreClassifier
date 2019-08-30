## Looperman Dataset

- Loop data were crawled from ```https://www.looperman.com/loops```, started from June to August 2019.

- We crawled the data in order of the **genres**. Please refer to ```genres.txt``` to check all the genres.
- The index of the folder name **_g(**  index **)** in ```loop_files``` refers to line indices of ```genres.txt```, indicating the genre of the dataset.
	- e.g. Loops in **_g1** refers to the genre **8Bit Chiptune**; **_g64** refers to the genre **Weird**.

- Issues:
	1. Loop#53311 & #67616 were corrupted.
	2. Loop#53281 & #36928 are monoaural files. 
	3. There may be some duplicate loops due to the continuous update of the website.