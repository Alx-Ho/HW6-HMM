import itertools

import numpy as np
import pytest

from hmm import HiddenMarkovModel


def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    model = HiddenMarkovModel(
        observation_states=mini_hmm["observation_states"],
        hidden_states=mini_hmm["hidden_states"],
        prior_p=mini_hmm["prior_p"],
        transition_p=mini_hmm["transition_p"],
        emission_p=mini_hmm["emission_p"],
    )

    obs_seq = mini_input["observation_state_sequence"]
    expected_hidden = mini_input["best_hidden_state_sequence"]

    forward_prob = model.forward(obs_seq)
    assert forward_prob > 0.0
    assert forward_prob <= 1.0 + 1e-12

    # brute-force forward probability for toy model
    obs_idx = [model.observation_states_dict[state] for state in obs_seq]
    num_states = len(model.hidden_states)
    brute_force = 0.0
    for path in itertools.product(range(num_states), repeat=len(obs_idx)):
        prob = model.prior_p[path[0]] * model.emission_p[path[0], obs_idx[0]]
        for t in range(1, len(obs_idx)):
            prob *= (
                model.transition_p[path[t - 1], path[t]]
                * model.emission_p[path[t], obs_idx[t]]
            )
        brute_force += prob
    assert np.isclose(forward_prob, brute_force, rtol=1e-6, atol=1e-12)

    viterbi_path = model.viterbi(obs_seq)
    assert len(viterbi_path) == len(obs_seq)
    assert np.array_equal(np.array(viterbi_path), expected_hidden)

    # edge case 1: empty observation sequence
    assert model.forward(np.array([])) == 0.0
    assert model.viterbi(np.array([])) == []

    # edge case 2: unknown observation state
    with pytest.raises(ValueError):
        model.forward(np.array(["__UNKNOWN__"]))
    with pytest.raises(ValueError):
        model.viterbi(np.array(["__UNKNOWN__"]))


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    model = HiddenMarkovModel(
        observation_states=full_hmm["observation_states"],
        hidden_states=full_hmm["hidden_states"],
        prior_p=full_hmm["prior_p"],
        transition_p=full_hmm["transition_p"],
        emission_p=full_hmm["emission_p"],
    )

    obs_seq = full_input["observation_state_sequence"]
    expected_hidden = full_input["best_hidden_state_sequence"]

    forward_prob = model.forward(obs_seq)
    assert forward_prob > 0.0
    assert forward_prob <= 1.0 + 1e-12

    viterbi_path = model.viterbi(obs_seq)
    assert len(viterbi_path) == len(obs_seq)
    assert np.array_equal(np.array(viterbi_path), expected_hidden)
