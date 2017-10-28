function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    
    % get the new hidden state
    hidden_state = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w,visible_data));
    
    %get the reconstruction
    reconstruction = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w,hidden_state));
    
    %get the reconstruction hidden state
    %reconstruction_hidden_state = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w,reconstruction));
    %Note that instead of sampling from the reconstructed state (as above), we can instead just use
    %the conditional probabilities. This is more effective because when we sample the reconstructed hidden state,
    %it simply increases the variance rather than the expected value of the gradient estimate.
    %when we use the conditional probabilities, it doesn't do this.
    reconstruction_hidden_state = visible_state_to_hidden_probabilities(rbm_w,reconstruction);
    
    
    % we want to decrease the energy of the initial visible / hidden state.
    % this gradient shows us how to do that    
    positive_gradient = configuration_goodness_gradient(visible_data,hidden_state);
    
    %we want to increase the energy of the state that is reached after the 
    % reconstruction - i.e. we don't want it to drift away from the data and the
    %state that gives rise to that data. The reconstruction and the reconstructed
    %hidden state is one that we don't want the system to learn to reproduce, in other words.
    negative_gradient = configuration_goodness_gradient(reconstruction,reconstruction_hidden_state);
    
    % the final gradient
    ret = positive_gradient - negative_gradient;
   
    
    
end
