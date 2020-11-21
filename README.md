# pugy
Revolutionary Online Education System : Fast Live Lesson Streams , Design with AI.

# Goals (Main Idea)
In this project, we planned to make revolutionary innovations in accessibility (our accesibility goal is smooth and speedless video streaming at low network bandwidths), which is the main issue of online education. We set out to ensure that students with low connection speeds can enter their education more easily and stay connected to the lesson without wasting time. For this, we aim to use the recently publised image compression algorithms that use the convolutional neural network method (The source of this algorithm is below.) , by implementing neural layers in the most performancely and minimalist way. This shows to what extent current technologies and algorithms can reduce playback problems in a video streams.

## About Project 
### Part 1 : Implementing Convolutional Neural Networks
Standard High Level AI libraries contain classes that do a lot of processing on memory and are too complex to perform well on low-end computers. This project contains of improved artificial neural networks layers with reliable memory management that do not go beyond the need of the corresponding compression algorithm. We wrote these implementations in neural folder and tested each layer with common train/test datas.

### Part 2 : Design Compress-Decompress Networks for Resizing  <According to this paper : [An End-to-End Compression Framework Based on Convolutional Neural Networks](https://arxiv.org/pdf/1708.00838v1.pdf) >
1) Compressing Method
![Compressing Method](https://hackernoon.com/hn-images/1*bjEWG34irHO7ZxllDGHqKQ.png)

2) Decompressing Method
![Decompressing Method](https://hackernoon.com/hn-images/1*bjEWG34irHO7ZxllDGHqKQ.png)

Recorded images are compressed and linked to videos derived from the same image format. Compressed images are decoded during playback.

### Part 3 : UI
By combining an application with all these powerful background features with the individual needs of the students, we have created an easy and useful design for the end user. 
