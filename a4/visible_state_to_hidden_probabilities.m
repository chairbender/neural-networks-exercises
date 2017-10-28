function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
    
    % get the logits for each hidden unit for each configuration
    % rbm_w is hidden x vis . We want hidden x num_configs.
    % visible state is vis x configs
    % so we do hidden x vis (rmb_w) * vis X configs (visible state) = hidden X configs
    z = rbm_w * visible_state;
    
    %probability of firing is sigmoid of the logit.
    hidden_probability = (1 ./ (1 .+ exp(-z)));     
end
