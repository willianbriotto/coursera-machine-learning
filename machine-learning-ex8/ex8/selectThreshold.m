function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions



    
    %Implementation Note: In order to compute tp, fp and fn, you may be able to use a 
    %vectorized implementation rather than loop over all the examples. This can be 
    %implemented by Octave/MATLAB’s equality test between a vector and a single number. 
    %If you have several binary values in an n-dimensional binary vector v ∈ {0,1}n, 
    %you can ﬁnd out how many values in this vector are 0 by using: sum(v == 0). 
    %You can also apply a logical and operator to such binary vectors. 
    %For instance, let cvPredictions be a binary vector of the size of your number of 
    %cross validation set, where the i-th element is 1 if your algorithm 
    %considers x(i) cv an anomaly, and 0 otherwise. You can then, for example, compute the 
    %number of false positives using: fp = sum((cvPredictions == 1) & (yval == 0)) 


    predictions = (pval < epsilon);

    %False positives
    tp = sum((predictions == 1) & (yval == 1));
    
    %False positives
    fp = sum((predictions == 1) & (yval == 0));
    
    %False negative
    fn = sum((predictions == 0) & (yval == 1)); 

    %You compute precision and recall by:
    %prec = tp / tp + fp
    %rec = tp / tp + fn

    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    
    %The F1 score is computed using precision (prec) and recall (rec):
    %F1 = (2 * prec * rec) /  (prec + rec)
    F1 = (2 * prec * rec) /  (prec + rec);
    
    
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
