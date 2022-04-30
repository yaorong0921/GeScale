# GeScale
More details about this work can be found in the paper [Talking with Your Hands: Scaling Hand Gestures and Recognition with CNNs](https://arxiv.org/pdf/1905.04225.pdf).

In this paper, 4 different models are used: 2D-SqueezeNet (version1.1), 2D-MobileNetV2; 3D-SqueezeNet (version1.1), 3D-MobileNetV2.

Training and testing the first two models can be found in the directory *GeScale_2D* and the other two are in *GeScale_3D*.


## SHGD (Scaling Hand Gesture Dataset)
You can download [SHGD](https://www.mmk.ei.tum.de/shgd/) here. The dataset includes two parts:
Single gestures and 3-tuple gestures. Every record includes infrared images and depth images. 

### Single gesture 
![](https://github.com/yaorong0921/GeScale/blob/master/example-single-gestures.gif)

### Gesture tuples
![](https://github.com/yaorong0921/GeScale/blob/master/example-3-tuple.gif)


### Citation
If you find this work useful or use the code, please cite as follows:

```
@inproceedings{kopuklu2019talking,
  title={Talking with your hands: Scaling hand gestures and recognition with cnns},
  author={Kopuklu, Okan and Rong, Yao and Rigoll, Gerhard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```
