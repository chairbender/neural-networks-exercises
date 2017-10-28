function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    
    % to calculate the energy of an RBM, we just add the weights of each
    % connected pair where both units are on. There are no biases, so the only
    % contribution to energy are pairs of activated, connected units.
    
    %first, get the logits for the hidden units
    z = rbm_w * visible_state; %z = hidden x configs
    
    %now zero out the values where the hidden units are off.
    %the contribution of the logit to the energy doesn't count if the units
    % which recieves that logit is off
    % z is the logits (hidden x configs). we have the hidden state (hidden x configs)
    % if we multiply each cell, the ones that are off will become zeroed out
    energy_per = z .* hidden_state; 
    
    %now we have energy_per, a matrix of hidden x configs. Each column 
    % represents a single configuration, and the values of that column repreent
    % the logits for hidden units which are active (0 for ones which aren't active)
    %We need to sum each column to get the energy per config.
    energy = sum(energy_per);
    
    %now we have a 1 x configs matrix, each cell indicating the energy of the 
    % config. We just need the average.
    
    G = (mean(energy));
    
end
