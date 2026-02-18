import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        obs = np.asarray(input_observation_states)
        if obs.size == 0:
            return 0.0
        if obs.ndim != 1:
            obs = obs.ravel()

        num_states = len(self.hidden_states)
        num_obs_states = len(self.observation_states)

        if self.prior_p.shape[0] != num_states:
            raise ValueError("prior_p size must match number of hidden states")
        if self.transition_p.shape != (num_states, num_states):
            raise ValueError("transition_p must be square with size N x N")
        if self.emission_p.shape != (num_states, num_obs_states):
            raise ValueError("emission_p must be N x M for N hidden and M observation states")

        try:
            obs_idx = np.array([self.observation_states_dict[state] for state in obs], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Unknown observation state: {exc.args[0]}") from exc

        # Step 2. Calculate probabilities with scaling to avoid underflow
        t_len = len(obs_idx)
        alpha = np.zeros((t_len, num_states), dtype=float)
        scales = np.zeros(t_len, dtype=float)

        alpha[0] = self.prior_p * self.emission_p[:, obs_idx[0]]
        scales[0] = alpha[0].sum()
        if scales[0] > 0:
            alpha[0] /= scales[0]

        for t in range(1, t_len):
            alpha[t] = (alpha[t - 1] @ self.transition_p) * self.emission_p[:, obs_idx[t]]
            scales[t] = alpha[t].sum()
            if scales[t] > 0:
                alpha[t] /= scales[t]

        # Step 3. Return final probability
        if np.any(scales == 0):
            return 0.0
        log_prob = np.sum(np.log(scales))
        return float(np.exp(log_prob))


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        obs = np.asarray(decode_observation_states)
        if obs.size == 0:
            return []
        if obs.ndim != 1:
            obs = obs.ravel()

        num_states = len(self.hidden_states)
        num_obs_states = len(self.observation_states)

        if self.prior_p.shape[0] != num_states:
            raise ValueError("prior_p size must match number of hidden states")
        if self.transition_p.shape != (num_states, num_states):
            raise ValueError("transition_p must be square with size N x N")
        if self.emission_p.shape != (num_states, num_obs_states):
            raise ValueError("emission_p must be N x M for N hidden and M observation states")

        try:
            obs_idx = np.array([self.observation_states_dict[state] for state in obs], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Unknown observation state: {exc.args[0]}") from exc

        t_len = len(obs_idx)
        log_prior = np.where(self.prior_p > 0, np.log(self.prior_p), -np.inf)
        log_trans = np.where(self.transition_p > 0, np.log(self.transition_p), -np.inf)
        log_emit = np.where(self.emission_p > 0, np.log(self.emission_p), -np.inf)

        # store probabilities of hidden state at each step
        viterbi_table = np.full((t_len, num_states), -np.inf, dtype=float)
        # store best path for traceback
        backpointers = np.full((t_len, num_states), -1, dtype=int)

        viterbi_table[0] = log_prior + log_emit[:, obs_idx[0]]

        # Step 2. Calculate Probabilities
        for t in range(1, t_len):
            for j in range(num_states):
                scores = viterbi_table[t - 1] + log_trans[:, j]
                best_state = int(np.argmax(scores))
                viterbi_table[t, j] = scores[best_state] + log_emit[j, obs_idx[t]]
                backpointers[t, j] = best_state

        # Step 3. Traceback
        if np.all(~np.isfinite(viterbi_table[-1])):
            return []
        best_last_state = int(np.argmax(viterbi_table[-1]))
        best_path_idx = np.zeros(t_len, dtype=int)
        best_path_idx[-1] = best_last_state
        for t in range(t_len - 1, 0, -1):
            best_path_idx[t - 1] = backpointers[t, best_path_idx[t]]

        # Step 4. Return best hidden state sequence
        return [self.hidden_states_dict[idx] for idx in best_path_idx]
