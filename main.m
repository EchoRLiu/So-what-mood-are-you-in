close all; clear all; clc

% Load uncropped Yale faces dataset.

m=243; n=320; nw=122*160;
% remember the shape of images before and after wavelet.
uncropped_faces=zeros(77760,165);
% we have 15 subjects with 11 different faces.

file='/Users/yuhongliu/Downloads/yalefaces_uncropped/yalefaces/subject';
subtitle=[".centerlight",".glasses",".happy",".leftlight",".noglasses",".normal",".rightlight",".sad",".sleepy",".surprised",".wink"];

for i=1:15
    for j=1:11
        if i<=9
            file_title=append(file,string(0),string(i),subtitle(j));
        else
            file_title=append(file,string(i),subtitle(j));
        end
        pos=11*(i-1)+j;
        uncropped_faces(:,pos)=reshape(imread(file_title), 77760, 1);
    end
end

%%

% As we can see, with wavelet and SVD, it does not make it very easy to
% obtain the eigenface, which means it does not detect very distinguished
% features of the face. (Even it does recoginize the human shape.)

[a, b, uncropped_faces_wave] = dc_wavelet(uncropped_faces, m, n);
feature = 60;

[U,S,V]=svd(uncropped_faces_wave,0);
U=U(:,1:feature);

% The first four POD modes.
figure(1);
for j = 1:4
    % Uncropped faces.
    subplot(2,2,j);
    ut1=reshape(U(:,j), a, b); % The shape after wavelet.
    ut2=ut1(a:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Uncropped Eigenfaces with mode' num2str(j)]);
end


%%

shape=128;
% Face Detection Step.
cropped = face_detection(uncropped_faces,m,n,shape); % The result will be double.
% Obtain wavelet version.
[a,b,cropped_wave] = dc_wavelet(cropped, shape, shape);

%%
[U1,S1,V1]=svd(cropped_wave,0);

% feature = 60;
% U1=U1(:,1:feature);

% The first six POD modes.
figure(1);
for j = 1:6
    % Uncropped faces.
    subplot(3,2,j);
    ut1=reshape(U1(:,j), a, b); % The shape after wavelet.
    ut2=ut1(a:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Uncropped Eigenfaces with mode' num2str(j)]);
end

% It's interetsing to see how much difference cropping the picture makes
% for feature extraction on human faces; it's also worth noticing that the
% features are still weaker than the features we obtained from original
% cropped dataset - this might be because compare to the cropped dataset,
% uncropped face dataset is rather small and not very diversed; extra
% thing is also added to the face like glasses.

%%

% After demonstrating the necessarity and ability of face detection
% (cropping faces), we now start to focus on how to classify different
% facial expressions.

close all; clear all; clc

% Load Mood Dataset.

m=48; n=48; mn= 48*48;

% Training Set.
training_set = zeros(mn, 3052);
% We have 7 different expressions and each of them has 436 images.
file_path='/Users/yuhongliu/Downloads/Moods/train/';
subpath = ["angry","disgust","fear","happy","neutral","sad","surprise"];

pos=0;
for i = 1:7
    full_path=append(file_path,subpath(i));
    file_ls=dir(full_path); % Get all the files info in the folder.
    for j = 1:436
        pos=pos+1;
        training_set(:,pos)=reshape(imread(fullfile(full_path, file_ls(j+2).name)), mn, 1);
    end
end

% Testing Set.
testing_set = zeros(mn, 777); % We have 111 images in 7 folders.
file_path='/Users/yuhongliu/Downloads/Moods/validation/';
subpath = ["angry","disgust","fear","happy","neutral","sad","surprise"];

pos=0;
for i = 1:7
    full_path=append(file_path,subpath(i));
    file_ls=dir(full_path); % Get all the files info in the folder.
    for j = 1:111
        pos=pos+1;
        testing_set(:,pos)=reshape(imread(fullfile(full_path, file_ls(j+2).name)), mn, 1);
    end
end

%%

tnt = [training_set testing_set];
[a,b,tnt_wave]=dc_wavelet(tnt,m,n);
[U,S,V] = svd(tnt_wave,0);

%%

% The first six POD modes.
figure(1);
for j = 1:6
    % Uncropped faces.
    subplot(3,2,j);
    ut1=reshape(U(:,j), a, b); % The shape after wavelet.
    ut2=ut1(a:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Uncropped Eigenfaces with mode' num2str(j)]);
end

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Normal scale')
subplot(2,1,2);
semilogy(diag(S),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Log scale')
% Seems like rank is 1.

figure(3);
for j=1:4
    subplot(4,1,j);
    plot(1:40,V(1:40,j),'ko-');
    xlabel('Mode')
    title(['first 40 Cropped images Projection onto mode' num2str(j)])
end

figure(4)
plot3(V(1:436,1), V(1:436, 2), V(1:436, 3), 'ko')
hold on
plot3(V(437:872,1), V(437:872, 2), V(437:872, 3), 'ro')
hold on
plot3(V(873:1308,1), V(873:1308, 2), V(873:1308, 3), 'go')
hold on
plot3(V(1309:1744,1), V(1309:1744, 2), V(1309:1744, 3), 'bo')
hold on
plot3(V(1745:2180,1), V(1745:2180, 2), V(1745:2180, 3), 'co')
hold on
plot3(V(2181:2616,1), V(2181:2616, 2), V(2181:2616, 3), 'yo')
hold on
plot3(V(2617:3052,1), V(2617:3052, 2), V(2617:3052, 3), 'mo')
hold on
legend('angry','disgust','fear','happy','neutral','sad','surprise'),
title('Visualisation of clusters');

%%

% Cross-validation!!
feature=[10 20 30 40 50 60 70 80 90];
cross=[300 500 1000];
feature_tb=[];

for e = 8% 1:9
    for w = 1%1:3
        avg_correct=0;
        for i=1:cross(w)

            % Seperate training and testing dataset.
            q1=randperm(547); q2=randperm(547); q3=randperm(547); q4=randperm(547);
            q5=randperm(547); q6=randperm(547); q7=randperm(547);
            t_angry=[V(1:436,1:feature(e));V(3053:3163,1:feature(e))];
            t_disgust=[V(437:872,1:feature(e));V(3164:3274,1:feature(e))];
            t_fear=[V(873:1308,1:feature(e));V(3275:3385,1:feature(e))];
            t_happy=[V(1309:1744,1:feature(e));V(3386:3496,1:feature(e))];
            t_neutral=[V(1745:2180,1:feature(e));V(3497:3607,1:feature(e))];
            t_sad=[V(2181:2616,1:feature(e));V(3608:3718,1:feature(e))];
            t_surprise=[V(2617:3052,1:feature(e));V(3719:3829,1:feature(e))];
            train=[t_angry(q1(1:436),:);t_disgust(q2(1:436),:);t_fear(q3(1:436),:);...
                t_happy(q4(1:436),:);t_neutral(q5(1:436),:);t_sad(q6(1:436),:);t_surprise(q7(1:436),:)];
            test=[t_angry(q1(437:end),:);t_disgust(q2(437:end),:);t_fear(q3(437:end),:);...
                t_happy(q4(437:end),:);t_neutral(q5(437:end),:);t_sad(q6(437:end),:);t_surprise(q7(437:end),:)];

            ctrain=[ones(436, 1); 2*ones(436,1); 3*ones(436,1); 4*ones(436,1); ...
                5*ones(436,1); 6*ones(436,1); 7*ones(436,1)];
            % Naive Bayesian method.
            % The method is not very good.
            % ########## .2630  ########
            % nb=fitcnb(train, ctrain);
            % pre=nb.predict(test);

            % Linear Discrimination.
            % ####### .2677 #########
            % pre=classify(test, train, ctrain);

            % SVM.
            % ######## .2629 ########
            % It's also very slow.
            % svm=fitcecoc(train,ctrain);
            % pre=predict(svm, test);
            
            % Quadratic DA classifier.
            % ####### .2712 #########
            % da=ClassificationDiscriminant.fit(train, ctrain);
            % pre=da.predict(test);
            
            % KNN.
            % ####### .2043 #########
            knn = ClassificationKNN.fit(train,ctrain);
            knn.NumNeighbors = 2;
            pre = knn.predict(test);
            
            % Tree Classification.
            % ##### .2046 ########
            tree = ClassificationTree.fit(train, ctrain);
            pre = tree.predict(test);

            bar(pre);

            correct=0;
            for k=1:7
                for j=((k-1)*111+1):(k*111)
                    if pre(j,1)==k
                        correct=correct+1;
                    end
                end
            end

            avg_correct = avg_correct + correct/777;

        end
        disp(avg_correct/cross(w));
        %feature_tb(w,e)=avg_correct/cross(w);
        disp('finish');
    end
end

% We need pre-processing to further extract facial expression features!!
% Also looking through all the images, some of them are not very
% representing.

%%

% Extracting feartues example with Blob detection.
I=uint8(reshape(training_set(:,7), 48,48));
corners=detectFASTFeatures(I);
strongest=selectStrongest(corners, 10);
[hog, validPoints, ptVis]=extractHOGFeatures(I, strongest);
imshow(I), hold on, plot(ptVis,'Color','Green');

%%

disgust_features=feature_extraction(training_set(:,437:872),m,n,463);
% Turns out we only have 105 good images with five strong features.
disgust_features_test=feature_extraction(testing_set(:,112:222),m,n,111);
% Turns out we only have 27 good images with five strong features.

%%

close all; clear all; clc
% Reload all images to select those ones with 5 best features.

m=48; n=48; mn= 48*48; mn_feature=180;

% Training Set.
training_features = zeros(mn_feature, 735);
% We have 7 different expressions and each of them has 105 images.
file_path='/Users/yuhongliu/Downloads/Moods/train/';
subpath = ["angry","disgust","fear","happy","neutral","sad","surprise"];

for i = 1:7
    full_path=append(file_path,subpath(i));
    file_ls=dir(full_path); % Get all the files info in the folder.
    temp=[];
    for j = 3:length(file_ls)
        temp(:,j-2)=reshape(imread(fullfile(full_path, file_ls(j).name)), mn, 1);
    end
    training_features(:,((i-1)*105+1):(105*i))=feature_extraction(temp,m,n,105);
end

%%

% Testing Set.
testing_features = zeros(mn_feature, 189); % We have 27 images in 7 folders.
file_path='/Users/yuhongliu/Downloads/Moods/validation/';
subpath = ["angry","disgust","fear","happy","neutral","sad","surprise"];

for i = 1:7
    full_path=append(file_path,subpath(i));
    file_ls=dir(full_path); % Get all the files info in the folder.
    temp=[];
    for j = 3:length(file_ls)
        temp(:,(j-2))=reshape(imread(fullfile(full_path, file_ls(j).name)), mn, 1);
    end
    testing_features(:,((i-1)*27+1):(27*i))=feature_extraction(temp,m,n,27);
end

%%

tnt = [training_features testing_features];
[U,S,V] = svd(tnt,0);

%%

a=5;b=36;
% The first six POD modes.
figure(1);
for j = 1:6
    % Uncropped faces.
    subplot(3,2,j);
    ut1=reshape(U(:,j), a, b); % The shape after wavelet.
    ut2=ut1(a:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['HOG features with mode' num2str(j)]);
end

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Normal scale')
subplot(2,1,2);
semilogy(diag(S),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Log scale')
% Seems like rank is 1.

figure(3);
for j=1:4
    subplot(4,1,j);
    plot(1:40,V(1:40,j),'ko-');
    xlabel('Mode')
    title(['first 40 HOG features Projection onto mode' num2str(j)])
end

figure(4)
train_ln=105
plot3(V(1:train_ln,1), V(1:train_ln, 2), V(1:train_ln, 3), 'ko')
hold on
plot3(V((train_ln+1):(2*train_ln),1), V((train_ln+1):(2*train_ln), 2), V((train_ln+1):(2*train_ln), 3), 'ro')
hold on
plot3(V((2*train_ln+1):(3*train_ln),1), V((2*train_ln+1):(3*train_ln), 2), V((2*train_ln+1):(3*train_ln), 3), 'go')
hold on
plot3(V((3*train_ln+1):(4*train_ln),1), V((3*train_ln+1):(4*train_ln), 2), V((3*train_ln+1):(4*train_ln), 3), 'bo')
hold on
plot3(V((4*train_ln+1):(5*train_ln),1), V((4*train_ln+1):(5*train_ln), 2), V((4*train_ln+1):(5*train_ln), 3), 'co')
hold on
plot3(V((5*train_ln+1):(6*train_ln),1), V((5*train_ln+1):(6*train_ln), 2), V((5*train_ln+1):(6*train_ln), 3), 'yo')
hold on
plot3(V((6*train_ln+1):(7*train_ln),1), V((6*train_ln+1):(7*train_ln), 2), V((6*train_ln+1):(7*train_ln), 3), 'mo')
hold on
legend('angry','disgust','fear','happy','neutral','sad','surprise'),
title('Visualisation of clusters');

%%

% Cross-validation!!
feature=[10 20 30 40 50 60 70 80 90 100];
cross=[300 500 1000];
feature_tb=[];
train_ln=105; test_ln=27;

for e = 10% 1:9
    for w = 1%1:3
        avg_correct=0;
        for i=1:cross(w)

            % Seperate training and testing dataset.
            q1=randperm(132); q2=randperm(132); q3=randperm(132); q4=randperm(132);
            q5=randperm(132); q6=randperm(132); q7=randperm(132);
            t_angry=[V(1:105,1:feature(e));V(736:762,1:feature(e))];
            t_disgust=[V(106:210,1:feature(e));V(763:789,1:feature(e))];
            t_fear=[V(211:315,1:feature(e));V(790:816,1:feature(e))];
            t_happy=[V(316:420,1:feature(e));V(817:843,1:feature(e))];
            t_neutral=[V(421:525,1:feature(e));V(844:870,1:feature(e))];
            t_sad=[V(526:630,1:feature(e));V(871:897,1:feature(e))];
            t_surprise=[V(631:735,1:feature(e));V(898:924,1:feature(e))];
            train=[t_angry(q1(1:105),:);t_disgust(q2(1:105),:);t_fear(q3(1:105),:);...
                t_happy(q4(1:105),:);t_neutral(q5(1:105),:);t_sad(q6(1:105),:);t_surprise(q7(1:105),:)];
            test=[t_angry(q1(106:end),:);t_disgust(q2(106:end),:);t_fear(q3(106:end),:);...
                t_happy(q4(106:end),:);t_neutral(q5(106:end),:);t_sad(q6(106:end),:);t_surprise(q7(106:end),:)];

            ctrain=[ones(105, 1); 2*ones(105,1); 3*ones(105,1); 4*ones(105,1); ...
                5*ones(105,1); 6*ones(105,1); 7*ones(105,1)];
            % Naive Bayesian method.
            % The method is not very good.
            % ########## .2630  ########
            nb=fitcnb(train, ctrain);
            pre=nb.predict(test);

            % Linear Discrimination.
            % ####### .2677 #########
            % pre=classify(test, train, ctrain);

            % SVM.
            % ######## .2629 ########
            % It's also very slow.
            % svm=fitcecoc(train,ctrain);
            % pre=predict(svm, test);
            
            % Quadratic DA classifier.
            % ####### .1966 #########
            %da=ClassificationDiscriminant.fit(train, ctrain);
            %pre=da.predict(test);
            
            % KNN.
            % ####### .2043 #########
            %knn = ClassificationKNN.fit(train,ctrain);
            %knn.NumNeighbors = 2;
            %pre = knn.predict(test);
            
            % Tree Classification.
            % ##### .2046 ########
            %tree = ClassificationTree.fit(train, ctrain);
            %pre = tree.predict(test);

            bar(pre);

            correct=0;
            for k=1:7
                for j=((k-1)*27+1):(k*27)
                    if pre(j,1)==k
                        correct=correct+1;
                    end
                end
            end

            avg_correct = avg_correct + correct/189;

        end
        disp(avg_correct/cross(w));
        %feature_tb(w,e)=avg_correct/cross(w);
        disp('finish');
    end
end

%% 

% Try with only three dataset.

close all; clear all; clc
% Reload all images to select those ones with 7 best features.

m=48; n=48; mn= 48*48; mn_feature=36*7;

% Training Set.
training_features = [];
% We have 3 different expressions.
file_path='/Users/yuhongliu/Downloads/Moods/train/';
subpath = ["angry","happy","sad"];

for i = 1:3
    full_path=append(file_path,subpath(i));
    file_ls=dir(full_path); % Get all the files info in the folder.
    temp=[];
    for j = 3:length(file_ls)
        temp(:,j-2)=reshape(imread(fullfile(full_path, file_ls(j).name)), mn, 1);
    end
    training_features=[training_features feature_extraction(temp,m,n,600)];
end

%%

% Testing Set.
testing_features = [];
file_path='/Users/yuhongliu/Downloads/Moods/validation/';
subpath = ["angry","happy","sad"];

for i = 1:3
    full_path=append(file_path,subpath(i));
    file_ls=dir(full_path); % Get all the files info in the folder.
    temp=[];
    for j = 3:length(file_ls)
        temp(:,(j-2))=reshape(imread(fullfile(full_path, file_ls(j).name)), mn, 1);
    end
    testing_features=[testing_features feature_extraction(temp,m,n,100)];
end

%%

tnt = [training_features testing_features];
[U,S,V] = svd(tnt,0);

%%

a=7;b=36;
% The first six POD modes.
figure(1);
for j = 1:6
    % Uncropped faces.
    subplot(3,2,j);
    ut1=reshape(U(:,j), a, b); % The shape after wavelet.
    ut2=ut1(a:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['HOG features with mode' num2str(j)]);
end

figure(2)
% Dominating features.
subplot(2,1,1); % normal scale.
plot(diag(S),'ko','Linewidth',[2]);
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Normal scale')
subplot(2,1,2);
semilogy(diag(S),'ko','Linewidth',[2]); % Log scale.
set(gca,'Fontsize',[14],'Xlim',[0 100]);
xlabel('Mode'), ylabel('diag(S): variance')
title('Log scale')
% Seems like rank is 1.

figure(3);
for j=1:4
    subplot(4,1,j);
    plot(1:40,V(1:40,j),'ko-');
    xlabel('Mode')
    title(['first 40 HOG features Projection onto mode' num2str(j)])
end

figure(4)
train_ln=600;
plot3(V(1:train_ln,1), V(1:train_ln, 2), V(1:train_ln, 3), 'ko')
hold on
plot3(V((train_ln+1):(2*train_ln),1), V((train_ln+1):(2*train_ln), 2), V((train_ln+1):(2*train_ln), 3), 'ro')
hold on
plot3(V((2*train_ln+1):(3*train_ln),1), V((2*train_ln+1):(3*train_ln), 2), V((2*train_ln+1):(3*train_ln), 3), 'go')
legend('angry','happy','surprise'),
title('Visualisation of clusters');

%%

% Cross-validation!!
feature=[10 20 30 40 50 60 70 80 90 100];
cross=[300 500 1000];
feature_tb=[];
train_ln=600; test_ln=100;

for e = 10% 1:9
    for w = 1%1:3
        avg_correct=0;
        for i=1:cross(w)

            % Seperate training and testing dataset.
            q1=randperm(train_ln+test_ln); q2=randperm(train_ln+test_ln); q3=randperm(train_ln+test_ln);
            t_angry=[V(1:train_ln,1:feature(e));V((3*train_ln+1):(3*train_ln+test_ln),1:feature(e))];
            t_happy=[V((train_ln+1):(2*train_ln),1:feature(e));V((3*train_ln+test_ln+1):(3*train_ln+2*test_ln),1:feature(e))];
            t_surprise=[V((2*train_ln+1):(3*train_ln),1:feature(e));V((3*train_ln+test_ln*2+1):(3*train_ln+3*test_ln),1:feature(e))];
            train=[t_angry(q1(1:train_ln),:);t_happy(q2(1:train_ln),:);t_surprise(q3(1:train_ln),:)];
            test=[t_angry(q1((train_ln+1):end),:);t_happy(q2((train_ln+1):end),:);t_surprise(q3((train_ln+1):end),:)];

            ctrain=[ones(train_ln, 1); 2*ones(train_ln,1); 3*ones(train_ln,1)];
            % Naive Bayesian method.
            % The method is the best.
            % ########## .5008  ########
            nb=fitcnb(train, ctrain);
            pre=nb.predict(test);

            % Linear Discrimination.
            % ####### .4806 #########
            % pre=classify(test, train, ctrain);

            % SVM.
            % ######## .2629 ########
            % It's also very slow.
            % svm=fitcecoc(train,ctrain);
            % pre=predict(svm, test);
            
            % Quadratic DA classifier.
            % ####### .4815 #########
            % da=ClassificationDiscriminant.fit(train, ctrain);
            % pre=da.predict(test);
            
            % KNN.
            % ####### .4719 #########
            %knn = ClassificationKNN.fit(train,ctrain);
            %knn.NumNeighbors = 2;
            %pre = knn.predict(test);
            
            % Tree Classification.
            % ##### .2046 ########
            %tree = ClassificationTree.fit(train, ctrain);
            %pre = tree.predict(test);

            bar(pre);

            correct=0;
            for k=1:3
                for j=((k-1)*test_ln+1):(k*test_ln)
                    if pre(j,1)==k
                        correct=correct+1;
                    end
                end
            end

            avg_correct = avg_correct + correct/(3*test_ln);

        end
        disp(avg_correct/cross(w));
        %feature_tb(w,e)=avg_correct/cross(w);
        disp('finish');
    end
end


%%

% Should try with different corner detection.

% imgaes input should be uint8().
function feature_vector=feature_extraction(images, m, n, limit) %,method) 
    [p,q]=size(images);
    images=uint8(images);
    feature_vector=[];
    
    % FAST corners method.
    pos=0;
    for i=1:q
        I=reshape(images(:,i), m,n);
        corners=detectFASTFeatures(I);
        feature=7;
        strongest=selectStrongest(corners, feature);
        [hog, validPoints]=extractHOGFeatures(I, strongest);
        [pass, gar]=size(hog);
        if (pass>=feature)
           pos=pos+1;
           feature_vector(:,pos)=reshape(hog(1:feature,1:36), 36*feature, 1);
        end
        if pos==limit
            disp('reach the limit!')
            break
        end
    end
end

%%

function [a,b,dcData] = dc_wavelet(dcfile, m, n)
    [p,q]=size(dcfile);
    nbcol = size(colormap(gray),1);
    
    % Obtain the shape after wavelet.
    X=double(reshape(dcfile(:,1),m,n));
    [cA,cH,cV,cD]=dwt2(X,'haar');
    cod_cH1 = wcodemat(cH,nbcol);
    cod_cV1 = wcodemat(cV,nbcol); 
    cod_edge=cod_cH1+cod_cV1;
    
    [a, b] = size(cod_edge);
    nw = a*b;
    dcData=zeros(nw,q);
    
    dcData(:,1)=reshape(cod_edge,nw,1);
    
    for i = 2:q
        X=double(reshape(dcfile(:,i),m,n));
        [cA,cH,cV,cD]=dwt2(X,'haar');
        cod_cH1 = wcodemat(cH,nbcol);
        cod_cV1 = wcodemat(cV,nbcol); 
        cod_edge=cod_cH1+cod_cV1; 
        dcData(:,i)=reshape(cod_edge,nw,1);
    end
end

%%

% This will return double datatype.
function cropped = face_detection(uncropped, m, n,shape)
    [p,q] = size(uncropped);
    cropped = zeros(shape*shape,q); 
    % Shape specify the image to be shape by shape, such as 128x resolution.
    % Create a cascade detector object.
    faceDetector = vision.CascadeObjectDetector();
    
    for i=1:q
        frame=uint8(reshape(uncropped(:,i), m,n));
        bbox=step(faceDetector, frame);
        frame=insertShape(frame, 'Rectangle',bbox);
        frame=imresize(imcrop(frame,bbox), [shape shape]);
        cropped(:,i)=reshape(frame(:,:,1), shape*shape,1);
    end
end
