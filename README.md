# So What Mood Are You In
This is a tutorial on how to use supervised learning to classify people's facial expression.

**Abstract**

As we can see in the Eigenfaces example \cite{eigenfaces}, PCA is capable of dealing with different lighting settings and extracting features very well. However, simply applying PCA on uncropped images, meaning the human face is not centered around the same area, gives very poor result. Viola-Jones algorithm \cite{ViolaJones} is presented here to show the way to detect human face and crop the image. A second part we hope to achieve is to use supervised learning to classify human facial expressions. It's first shown that by using SVD and Haar Wavelet to extract the features, the result of classification is not ideal. Histogram of oriented gradients (HOG) feature extraction method \cite{hog} is then used to try to extract shapes of eyes, nose, eyebrows and mouth, based on which, seven types of facial expressions will be classified.

**Introduction and Overview**

First, we use the Yale uncropped face data sets \cite{yaleuncroppedface} to illustrate how Viola-Jones algorithm is used to detect human face and crop the image around it. We then use a facial expression dataset \cite{facialexpressionds} to illustrate a way to classify facial expressions.

**Problem Description**

**.1Yale Uncropped Face**
After using haar wavelet to extract features from the images, the eigenface we got from SVD is not showing any facial features, compared to the result from Yale Cropped Face dataset. We need to find a way to detect the area human face is centered around, and also find the edge location of that area for future cropping or feature extraction from that area.

**.2Facial Expression**
In the facial expression dataset, we have seven different types of facial expressions, such as angry, disgust, fear, happy, neutral, sad, and surprise, and each of the image is being cropped around the face. However, they are not posed images, meaning the human is not generally facing the camera and they could have different postures. Another problem spotted by looking through the images is that some of the facial expressions are very difficult to tell even by human eyes, meaning one image could be angry or disgust.

**General Approach**

**.1Yale Uncropped Face**

We first show the SVD result of uncropped faces dataset with Haar wavelet does not give any facial features; we then use Viola-Jones algorithm to detect face from every image and crop the images around the face area. We then use Haar wavelet and SVD to obtain the eigenface to see if we have good features.

**.2Facial Expression**

We first show the features with eigenface of the facial expression through Haar wavelet and SVD, and try to train and test on them. By examining the images among seven classes, it's interesting to notice that the main difference between different facial expressions is the shape of the person's eyes, mouth and the muscles around their eyebrows. For instance, when a person is angry, their eyesbrows are normally frowned; when a person is happy, their mouth typically has a smiling shape. Observing this, we then try to implement HOG to extract the shape and direction of muscles on human's face, and use those features to train and test the classification.

**Theoretical Background**
As we learned from our textbook \cite{kutz_2013} and illustrated in the Eigenface project \cite{eigenfaces}, Principle Component Analysis is a very useful way to extract features. Generally, if $A_{mn}$ is a matrix containing n data as column vector with length m, and $n_i (i=1,...,k,\Sigma_{i=1}^{k}n_i = n)$ of the columns represent class i data, we can decompose it as:

        A_{mn}=\hat{U}\hat{\Sigma}V*.

where first few parts of $\hat{U}$ can be used to represent the whole dataset's features, the diagonal of $\hat{\Sigma}$ represent how much each mode explains the variance of the dataset, and $V$ is the representation of each datapoint and $XV$ are the projection of the data onto each variable. In this project, we can analyze $\Sigma$ to see how many modes can give us the most information, and use V to train and test the classification work.

Used in almost all computer vision work nowadays, Viola-Jones algorithm \cite{ViolaJones} is a very useful way to detect human face area in a single image or video frames, based on which so many feature extraction methods are made possible. First, Haar wavelet is used to extract the the edges of the images; second, an integral value board is built as the basic combinations of pixels to save computing time later (we get it by adding or subtracting the values from the first pixel to the last pixel sequentially and store them in the table), and it will be used later to compute more combinations of the pixels; third, adaboost tree and cascade classifier are used to decide if a region has face or not (adaboost tree has one node and two child nodes and some of trees have larger decision power than others; based on the Haar wavelet features, the classifier can decide whether a region is part of the face or not; those regions with faces then have larger decision power since other larger or neighbor region can be based on them).

HOG feature \cite{hog_paper} is a classic way to describe features or shapes of objects in images. The main idea is, by calculating gradients of pixels, we can obtain the directions of edges, thus describing the objects. The gradients of x and y directions between a pixel and other pixels around it would first be calculated, together giving us the orientation of those gradients; the histogram of those gradients is then calculated to dense all the information.

A few supervised learning methods used are also introduced here. K Nearest Neighnours (KNN) classifies each data point based on its closest k neighbours. Each neighbour has deciding power of which class the data point belongs to, based on the distance between the data point and the neighbour. Naive Bayes (NB) uses the idea of conditional probability and Bayes rules. For instance, if we have two classes 0 and 1,

        \frac{p(0|x)}{p(1|x)}=\frac{p(x|0)p(0)}{p(x|1)p(1)}.

now all the probability on the right hand can be calculated, which tells us which class the data point belongs to with a larger probability. Linear Discrimination Analysis (LDA) finds the optimal way to project the data onto a line to maximize the distance between the projected data from different classes. Support Vector Machine (SVM) is a method that tries to find the optimal line to separate the data, where the curves along each data class's boundary points are optimized first, and another set of lines are optimized to maximize the distance between every two curves we found before. An advantage of SVM is that it can produce non-linear curve separating the data. Classification Trees separate data by proposing many questions and based on the yes or no answers.

**Algorithm Implementation and Development**
The general approach is implemented in the following way (all detailed explanation of the code is in the comment along the code).

**.1Yale Uncropped Face

Yale Uncropped Face data set is first loaded into \texttt{uncropped faces}

Viola-Jones algorithm is then applied to all the images to detect faces and crop all images using Algorithm~\ref{alg:viola-jones} and store the result as \texttt{cropped}

    \STATE{Obtain the matrix \texttt{uncropped faces} where each column represent one image, the size of each image as \texttt{m, n} and the \texttt{shape} that defining the cropped image's resolution}
    \STATE{Obatin the size of \texttt{uncropped faces} as \texttt{p, q}}
    \STATE{Initialize \texttt{vision.CascadeObjectDetector()} as \texttt{faceDetector}}
    \FOR{$j = 1:q$}
        \STATE{Obtain column j of \texttt{uncropped faces}, reashape them to m by n, change the datatype to uint8 and store the result in \texttt{frame}.}
        \STATE{apply \texttt{step(faceDetector, frame)} and store the result as \texttt{bbox}}
        \STATE{crop \texttt{frame} by \texttt{imcrop(frame, bbox)} and resize to shape by shape resolution. Store the result into \texttt{frame}}
        \STATE{reshape \texttt{frame} into a column vector and store it into the column j of \texttt{cropped}}
    \ENDFOR
    
Haar wavelet is then applied to all images in \texttt{cropped} to extract edge features using Algorithm~\ref{alg:wavelet} and store the result into \texttt{cropped wave}
    
    \STATE{Obtain the matrix \texttt{uncropped faces} where each column represent one image, the size of each image as \texttt{m, n}}
    \STATE{Obatin the size of \texttt{uncropped faces} as \texttt{p, q}}
    \STATE{Get the column size of \texttt{colormap(gray)} as \texttt{nbcol}}
    \FOR{$j = 1:q$}
        \STATE{Obtain column j of \texttt{uncropped faces}, reashape them to m by n, change the datatype to double and store the result in \texttt{X}.}
        \STATE{apply \texttt{dwt2(X, 'haar')} and store the results as \texttt{cA, cH, cV, cD}}
        \STATE{apply \texttt{wcodemat(,nbcol)} to both \texttt{cH, cV} and store results as \texttt{cod cH1, cod cV1}}
        \STATE{Store \texttt{cod cH1 + cod cV1} as \texttt{cod edge}}
        \STATE{reshape \texttt{cod edge} into a column vector and store it into the column j of \texttt{dcData}}
    \ENDFOR
    
SVD is then applied to \texttt{cropped wave} and obtain results \texttt{U, S, V}

\texttt{U, S, V} are then analyzed in the next section.

**.2Facial Expression

Choose to use Haar wavelet or HOG feature extraction.

For Haar wavelet method: load facial expression training and testing datset as \texttt{training set, testing set}; combine \texttt{training set, testing set} as \texttt{tnt} and apply Haar wavelet using Algorithm~\ref{alg:wavelet}, and store the result as \texttt{tnt wave}.
    
or HOG feature extraction, when loading the training and testing data set, we apply HOG feature extraction to each image using Algorithm~\ref{alg:hog}, combine the results and store as \texttt{tnt hog}.
    
    \STATE{Get inputs \texttt{images, m, n}}
    \STATE{Obtain the number of images we need to process as \texttt{q} and change the data type to uint8 for processing.}
    \STATE{Initialize \texttt{pos = 0}}
    \FOR{$j = 1:q$}
        \STATE{Obtain column j, reshape them to m by n, and store as I}
        \STATE{Get corner points \texttt{corners=detectFASTFeatures(I)}}
        \STATE{Choose the strongest feature points \texttt{strongest=selectStrongest(corners, feature=7)}}
        \STATE{extract HOG features using \texttt{extractHOGFeatures(I, strongest)} and store results as \texttt{hog, validPoints}}
        \STATE{reshape hog into a column vector and store it into \texttt{feature vector}}
    \ENDFOR
    
apply svd to \texttt{tnt wave} or \texttt{tnt hog} and obtain results as \texttt{U, S, V}

visualize \texttt{U, S, V}

use Algorithm~\ref{alg:crossval} on \texttt{V} with cross validation to obtain average rate of correct classification.

    \STATE{Initialize \texttt{avg correct} as $0$, \texttt{cross}, cross validation times, as $300$.}
    
    \FOR{$j = 1:cross$}
        \STATE{Within each class, shuffle data and separate them into \texttt{train} and \texttt{test}}
        \STATE{Create labels for \texttt{train} as \texttt{ctrain}}
        \STATE{Choose each of the following methods.}
        \STATE{For Naive Bayes method, apply \texttt{nb = fitcnb(train, ctrain)} and obtain \texttt{pre} using \texttt{nb.predict(test)}}
        \STATE{For Linear Discrimination method, obtain \texttt{pre} using \texttt{classify(test, train, ctrain)}}
        \STATE{For SVM method, apply \texttt{svm = fitcecoc(train, ctrain)} and obtain \texttt{pre} using \texttt{predict(svm, test)}}
        \STATE{For Quadratic Discrimination method, apply \texttt{da = ClassificationDiscriminant.fit(train, ctrain)} and obtain \texttt{pre} using \texttt{da.predict(test)}}
        \STATE{For KNN method, apply \texttt{knn = ClassificationKNN.fit(train, ctrain)}, define the number of neighbours we want using \texttt{knn.NumNeighbors = 2} (we use 2 here), and obtain \texttt{pre} using \texttt{knn.predict(test)}}
        \STATE{For Tree Classification method, apply \texttt{tree = ClassificationTree.fit(train, ctrain)} and obtain \texttt{pre} using \texttt{tree.predict(tree)}}
        \STATE{Visualize \texttt{pre} using \texttt{bar(pre)}}
        \STATE{calculate \texttt{correct} using Algorithm~\ref{alg:correct} and update \texttt{avg correct = avg correct + correct / total num of testing data}}
    \ENDFOR
    \STATE{Obtain the true \texttt{avg correct = avg correct/cross}}

    \STATE{Initialize \texttt{correct = 0}}
    \FOR{$j = 1:num of total classes$}
        \IF{results of \texttt{pre} for class j equal j}
        \STATE{add number of correct classification to \texttt{correct}}
        \ENDIF
    \ENDFOR

**Computational Results

**.1Yale Uncropped Face

As we can see in the right side of Figure~\ref{fig:bad_eigenfaces}, with wavelet and SVD, it's not very easy to detect distinguished features of the face, compare to the left side, but it does recognize the human shape. After cropping, in Figure~\ref{fig:after_crop} the result seems better. It's interetsing to see how much difference cropping the picture makes
for feature extraction on human faces; it's also worth noticing that the features are still weaker than the features we obtained from original cropped dataset - this might be because compare to the cropped dataset, uncropped face dataset is rather small and not very diversed and extra thing like glasses is also added to the face.

![figure 1](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/comparison.jpg)
![figure 2](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/after_crop.jpg)


**.2Facial Expression**

After demonstrating the necessity and ability of face detection(cropping faces), we now start to focus on how to classify different facial expressions. We first try to use Haar wavelet to extract the features, and use SVD on those features. In Figure~\ref{fig:Eigenfaces_tnt}, we can see the Eigenfaces of the seven classes. We can also see in Figure~\ref{fig:variance_tnt}, the rank is very low, meaning we can represent the data in a very low dimension way. But looking at the clusters of the seven classes in Figure~\ref{fig:clusters_tnt}, the training process will be very difficult.

![figure 3](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/Eigenfaces_tnt.jpg)
![figure 4](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/v_tnt.jpg)
![figure 5](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/variance_tnt.jpg)
![figure 6](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/clusters_tnt.jpg)

Among NB, LDA, SVM, QDA, KNN, Tree classification method, NB gives the best classification results. Table~\ref{tab:NB} illustrates how the rate of correct classification changes with the number of modes and the cross validation times we use. We can see from 300, increasing cross validation time does not change the rate that much; at first, increasing the modes improves the rate but not very significant after 80 modes. The results are not very good. We then try to use HOG to extract facial expression features to see if that makes a difference.

         modes      & $10$   & $20$ & $30$& $40$ & $50$& $60$& $70$& $80$& $90$\\
        \hline
        $cross=300$  & 0.21 & 0.25 & 0.26 &	0.26 & 0.26 & 0.26 & 0.26 & 0.27 & 0.26 \\
        $cross=500$  & 0.21 & 0.26 & 0.26 &	0.27 & 0.26 & 0.26 &  0.26 & 0.26 & 0.26 \\
        $cross=1000$  & 0.21 & 0.26 & 0.26 & 0.26 &	0.26 & 0.26 &  0.26 & 0.27 & 0.26 \\
    
A thing about the data set worth noticing is that in some images, the human is not facing the camera, it's not staged, and some of them are not very representing. So we can try to use HOG features detection to try to select those images that are more representing to train the classifier. Here is an example of HOG detected features Figure~\ref{fig:cornerhog_happy}. In Figure~\ref{fig:feature_cluster}, we observe the same clusters behavior, which means we might also have a bad classification results. We tried it anyway, and the results are similar to Table~\ref{tab:NB}.

![figure 7](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/cornerhog_happy_eg.jpg)
![fugure 8](https://github.com/EchoRLiu/So-what-mood-are-you-in/blob/master/feature_cluster.jpg)
    
We also try to reduce the number of classes to three (angry, happy, surprise). The average correct classification rate using NB and HOG feature extraction is $.5008$.

% Summary and Conclusions
\section{Summary and Conclusions}
Through these two examples, we can first see the importance of face detection and alignment for good feature extraction, and the power of Viola-Jones algorithm. In the facial expression classification example, we can see how HOG feature extraction focuses on the directions or shapes of human eyes and mouths. The classification is not very successful, and more work and research need to be done to better classify seven classes of facial expression with very diverse and spontaneous facial expression data set.

@misc{eigenfaces,
    title={Eigen Faces},
    url={https://github.com/EchoRLiu/Eigenfaces-Music-Genre-Classification},
    publisher={Github},
    author={Liu, Echo},
    year={2020},
    month={Mar}
}

@article{ViolaJones,
    author = {P. Viola and M. Jones},
    title = {Rapid object detection using a boosted cascade of simple features},
    year = {2001},
    publisher = {IEEE},
}

@article{hog,
    author={N. Dalal and B. Triggs},
    title={Histograms of oriented gradients for human detection},
    year={2005},
    journal={IEEE}
}

@misc{yaleuncroppedface,
        title={ExtYaleDatabase},
        url={http://vision.ucsd.edu/~leekc/ExtYaleDatabase/download.html},
        publisher={UCSB Vision Lab}}

@misc{facialexpressionds,
        title={Face Expression Recognition Dataset},
        url={https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset},
        publisher={Kaggle}}

@book{kutz_2013,
    place={Oxford}, 
    title={Data-driven modeling \& scientific computation: methods for complex systems \& big data},
    publisher={Oxford University Press},
    author={Kutz, Jose Nathan},
    year={2013}
}

@article{hog_paper,
    author={Dalal, N. and B. Triggs},
    title={Histograms of Oriented Gradients for Human Detection}, 
    publisher={IEEE},
    year={2005},
    month={June}
    }
