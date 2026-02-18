# HW6-HMM

Note: I used code LLMs for nice formatting and commenting. 

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





## Tasks and Data 
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)



For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences. 
 * `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 


Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Are the inputs in compatible shapes/sizes which each other? 
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases. 

Finally, please update your README with a brief description of your methods. 


## Methods

The Forward algorithm is implemented with per-timestep scaling to avoid underflow. Each row of the forward table is normalized by its sum, and the final likelihood is recovered by multiplying the scale factors (tracked in log-space for stability).

The Viterbi algorithm is implemented in log-space to avoid underflow. The dynamic program stores the best log-probability for each state at each timestep and a backpointer table for traceback, yielding the most likely hidden state sequence.



## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [x] complete the `forward` function in the HiddenMarkovModelClass <br>
  [x] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [x] Ensure functionality on mini and full weather dataset <br>
  [x] Account for edge cases 

[TODO] Packaging <br>
  [x] Update README with description of your methods <br>
  [x] pip installable module (optional)<br>
  [ ] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](https://forms.gle/xw98ZVQjaJvZaAzSA)

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
