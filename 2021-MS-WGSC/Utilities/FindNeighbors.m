function [ refPtchIdx, cCoarserImg ] = FindNeighbors( y, par )
% Find for each patch its neighbors.
% Namely, for each patch find all the other patches inside its searching window.

% Extract Data From struct
patchSize           = par.patsize;
stepSize            = par.step;
searchWinSize       = par.SearchWinSize;
winSizeMethod       = par.winSizeMethod;
downScaleFactor     = par.downScaleFactor;
% ------------------------------------------------

numScales = length(downScaleFactor);

if strcmp(winSizeMethod, 'ChangeWithScale')
    vSearchWinSize = ceil(searchWinSize .* downScaleFactor);
elseif strcmp(winSizeMethod, 'Fixed')
    vSearchWinSize = searchWinSize .* ones(numScales, 1);
    vStepSize = ones(numScales, 1);
    vStepSize(1) = stepSize;
else
    error('Parameter ''winSizeMethod'' has invalid value.');
end

cCoarserImg = cell(numScales,4); % dim1 - mNeighbors , dim2 - neighborsNum , dim3 - imgHeight , dim4 - imgWidth
for ii = 1:numScales
    mCoarserImg = imresize(y(:,:,1), downScaleFactor(ii));
    [coarserImgHeight, coarserImgWidth, ~] = size(mCoarserImg); % TO DO: calc coarser image size without using y
    [mCoarserNeighbors, coarserNeighborsNum, currRefPtchIdx] = GetNeighborIndex(coarserImgHeight,...
        coarserImgWidth, vStepSize(ii), vSearchWinSize(ii), patchSize);
    cCoarserImg{ii,1} = mCoarserNeighbors;
    cCoarserImg{ii,2} = coarserNeighborsNum;
    cCoarserImg{ii,3} = coarserImgHeight;
    cCoarserImg{ii,4} = coarserImgWidth;
    if ii==1, refPtchIdx = currRefPtchIdx; end % Take refPtchIdx from original scale
end

end