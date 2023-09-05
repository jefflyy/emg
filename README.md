# 肌电信号处理

基于myo腕带的手势分类，是对[一篇论文](https://doi.org/10.1109/AIM.2019.8868766)的复现。

### 依赖

* myo connect：蓝牙适配器的驱动。由于thalmic疑似倒闭(?)，官网已无法访问。
* myo-sdk：官方sdk。
* [myo-python](https://github.com/NiklasRosenstein/myo-python)：sdk的python接口。
* [scikit-learn](https://scikit-learn.org/stable/index.html)：提供一些分类算法。

### 运行

在连接myo之前，请仔细阅读[myomex的文档](https://github.com/mark-toma/MyoMex)，这能帮助解决很多问题。

执行`python main.py train`以录制动作和训练，执行`python main.py test`以测试。
