close all; clear all; clc

% Load uncropped faces dataset.
m2=243; n2=320; nw2=122*160;% remember the shape of images.
uncropped_faces=zeros(77760,165); % we have 15 subjects with 11 different faces.

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
% obtain the eigenface.
uncropped_faces_wave = dc_wavelet(uncropped_faces, m2, n2, nw2);
feature = 60;

[U,S,V]=svd(uncropped_faces_wave,0);
U=U(:,1:feature);

% The first four POD modes.
figure(1);
for j = 1:4
    % Uncropped faces.
    subplot(2,2,j);
    ut1=reshape(U(:,j), 122, 160);
    ut2=ut1(122:-1:1,:);
    pcolor(ut2);
    set(gca,'Xtick',[],'Ytick',[]);
    title(['Uncropped Eigenfaces with mode' num2str(j)]);
end
