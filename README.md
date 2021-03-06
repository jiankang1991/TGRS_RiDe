# Rotation Invariant Deep Embedding for Remote Sensing Images

---

[Jian Kang](https://github.com/jiankang1991), [Ruben Fernandez-Beltran](https://scholar.google.es/citations?user=pdzJmcQAAAAJ&hl=es), [Zhirui Wang](), [Xian Sun](http://people.ucas.ac.cn/~sunxian), [Jingen Ni](https://scholar.google.com/citations?hl=en&user=hqZB5wQAAAAJ&view_op=list_works&sortby=pubdate), [Antonio Plaza](https://www.umbc.edu/rssipl/people/aplaza/)

This repo contains the codes for the TGRS paper: [Rotation Invariant Deep Embedding for Remote Sensing Images](). We first propose a rule that the deep embeddings of rotated images should be closer to each other than those of any other images (including the images belonging to the same class). Then, we propose to maximize the joint probability of the leaveone-out image classification and rotational image identification. With the assumption of independence, such optimization leads to the minimization of a novel loss function composed of two terms: 1) a class-discrimination term, and 2) a rotation-invariant term. 

<p align="center">
<img src="framework.PNG" alt="drawing"/>
</p>

## Usage

`main.py` is the script of the proposed method for training and validation.

`utils/NCA_RI_Mul` contains the proposed loss function.


## Citation

```
@article{kang2021RiDe,
  title={{Rotation Invariant Deep Embedding for Remote Sensing Images}},
  author={Kang, Jian and Fernandez-Beltran, Ruben and Wang, Zhirui and Sun, Xian and Ni, Jingen and Plaza, Antonio},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  note={DOI:10.1109/TGRS.2021.3088398}
  publisher={IEEE}
}
```

