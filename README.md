I have reimplemented the class and functions originally developed by Jordan Van (https://github.com/jordanvaneetveldt/Kolmogorov-Face-Recognition
), a project that exemplifies an elegant and underappreciated approach to image classification.

In the contemporary landscape, the field overwhelmingly favors convolutional neural networks and massive pre-trained models. While powerful, these methods are fundamentally limited in low-data regimes, such as medical imaging tasks including cancer cell classification. In contrast, Kolmogorov complexity offers a theoretically principled alternative that excels precisely where data scarcity renders conventional deep learning approaches ineffective.


It must be emphasized that while the underlying concept is derived from established literature, notably an IEEE publication, the implementation itself is entirely Van’s innovation. This work deserves far greater attention within the research community, both for its intellectual elegance and its practical relevance, and it is imperative that it be disseminated more widely so that the full potential of compression-based classification can be appreciated and utilized.

### Usage
```
#### Install using # git clone URL

### cd working dir of the clone
### pip install .

from kolmogorov import CompressionKNN
```

