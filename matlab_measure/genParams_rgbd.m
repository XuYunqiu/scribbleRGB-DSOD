function p = genParams_rgbd()
%% Generate environment and user-specific parameters.
p.salObjSets = {'NJU2K_Test','STERE','DES','NLPR_Test','LFSD','SIP','SSD'};
p.GTsuffix = {'.png','.png','.png','.png','.png','.png','.png'};
p.Imgsuffix = {'.png','.png','.png','.png','.png','.png','.png'};

p.salObjSets = p.salObjSets(:);
setNum = length(p.salObjSets);
%% set p.algMapDir as your own saliency map directory
p.algMapDir = '../results/';
%% p.GTDir is ground truth saliency map directory
p.GTDir = '../dataset/test_data/gt/';  %% gt file path

%%
end