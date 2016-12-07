function [ segmentVector, segmentboundaries ] = intrinsic_seg( Z )
    [~, xn, ~] = size(Z);
    R = (triu(ones(xn,xn-1),1) - triu(ones(xn, xn-1))) + (triu(ones(xn, xn-1),-1)-triu(ones(xn, xn-1)));
    
    Z = normalize(Z);
    
    ZR = Z*R;

    g = gausswin(3);
    g = g/sum(g);

    y3 = conv(mean(abs(ZR)), g, 'same');

    y3 = y3 - mean(y3);
    y3(y3 < 0) = 0;
    
    t = 0.02;
    
    y3(y3 < t) = 0;

    [~,imax,~,~] = extrema(y3);

    segmentboundaries = sort(imax);

    segmentVector = zeros(xn,length(segmentboundaries)+1);
    previous = 0;
    for i=1:length(segmentboundaries)
        tempVec=zeros(1,length(segmentboundaries)+1);
        tempVec(i) = 1;
        tempVec = repmat(tempVec,segmentboundaries(i)-previous,1);
        segmentVector(1+previous:segmentboundaries(i),:) = tempVec;
        previous = segmentboundaries(i);
    end

end

