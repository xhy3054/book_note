# 第三章：TensorFlow入门
## 3.1 TensorFlow计算模型--计算图
<div style="text-align: center">
<img src="tensors_flowing.gif"/>
</div>

> tensorflow中的计算模型是一个计算图谱的形式，计算图谱上的每个节点都代表一个运算，节点上输入的边代表这个节点的运算要依赖于边源头节点的输出，节点上输出的边代表这个节点的运算的结果被别的运算依赖。如同上图展示的那样，在计算过程中，数据(tensor)在计算图谱的节点间流动(flow).

- 



