# RawNet2 ASVspoof 2021 baseline

By Hemlata Tak, EURECOM, 2021

------
# Basic Machine Learning 2025.1(Prof Vu Hai)
Change Made by Mai Viet Anh, Tran Phu Nghia, Nguyen Quang Dung, Vu Ngoc Dang Khoa

The code in this repository serves as one of the baselines of the ASVspoof 2021 challenge, using an end-to-end method that uses a model based on the RawNet2 topology as described [here](https://arxiv.org/abs/2011.01108).


### Dataset
Our model is trained on the logical access (LA) train  partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset?select=LA).

### Training
To train the model run:
```
python main.py --track=DF --loss=CCE   --lr=0.0001 --batch_size=32 --database_path "/your/path/to/data/ASVspoof_database/"  --protocols_path "/your/path/to/protocols/ASVspoof_database/"
```
### Testing
You can use test.py with an audio input file, the program will load the pre-trained model and decide whether the input is bonafide or spoof

## Citation
If you use this code in your research please use the following citation:
```bibtex
@INPROCEEDINGS{9414234,
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={End-to-End anti-spoofing with RawNet2}, 
  year={2021},
  pages={6369-6373}
}

```
