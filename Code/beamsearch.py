import numpy as np

# Node class for beam search.
class BeamNode(object):
    def __init__(self, parent, state, value, cost, extras):
        super(Node, self).__init__()
        self.value = value      # value/predicted word for current node.
        self.parent = parent    # parent Node, None for root
        self.state = state      # current node's lstm hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost  # cumulative cost of entire chain upto current node.
        self.length = 1 if parent is None else parent.length + 1    # length of entire chain
        self.extras = extras    # any extra variables to store for the node
        self._sequence = None   # to hold the node's entire sequence.

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [s.value for s in self.to_sequence()]

    def to_sequence_of_extras(self):
        return [s.extras for s in self.to_sequence()]

def beam_search(initial_state_function, generate_function, X, end_id, batch_size=1, beam_width=4, num_hypotheses=1, max_length=50, vocab_size=33):
    """Beam search for neural network sequence to sequence (encoder-decoder) models.
    :param initial_state_function: A function that takes X as input and returns state (2-dimensonal numpy array with 1 row
                                   representing decoder recurrent layer state - currently supports only one recurrent layer).
    :param generate_function: A function that takes X, Y_tm1 (1-dimensional numpy array of token indices in decoder vocabulary
                              generated at previous step) and state_tm1 (2-dimensonal numpy array of previous step decoder recurrent
                              layer states) as input and returns state_t (2-dimensonal numpy array of current step decoder recurrent
                              layer states), p_t (2-dimensonal numpy array of decoder softmax outputs) and optional extras
                              (e.g. attention weights at current step).
    :param X: List of input token indices in encoder vocabulary.
    :param start_id: Index of <start sequence> token in decoder vocabulary.
    :param end_id: Index of <end sequence> token in decoder vocabulary.
    :param beam_width: Beam size. Default 4.
    :param num_hypotheses: Number of hypotheses to generate. Default 1.
    :param max_length: Length limit for generated sequence. Default 50.
    """
    # if isinstance(X, list) or X.ndim == 1:
    #   X = np.array([X], dtype=np.int32).T
    # assert X.ndim == 2 and X.shape[1] == 1, "X should be a column array with shape (input-sequence-length, 1)"

    #initial_state, initial_value = initial_state_function(X, seq_sizes)
    initial_state, initial_value = initial_state_function(X, batch_size)
    next_fringe = [Node(parent=None, state=initial_state, value=initial_value, cost=0.0, extras=None)]
    hypotheses = []

    #entire_vocab = True if beam_width <=
    for step in range(max_length):
        fringe = []
        #print('n.value:', [n.value for n in next_fringe])
        for n in next_fringe:
            if (step != 0 and n.value == end_id) or step == max_length - 1:
                hypotheses.append(n)
            else:
                fringe.append(n)

        if not fringe or len(hypotheses) >= num_hypotheses:
        #if not fringe:
            #print('exiting.')
            #print('len(hypotheses):', len(hypotheses))
            break

        Y_tm1 = [n.value for n in fringe]
        state_tm1 = [n.state for n in fringe]
        #state_t, p_t, extras_t = generate_function(X, Y_tm1, state_tm1)
        state_t, p_t, extras_t = generate_function(Y_tm1, state_tm1)
        #print('raw pred:',p_t )
        #print('max raw pred:', np.argmax(p_t, axis=1))
        Y_t = np.argsort(p_t, axis=1)[:,-beam_width:] # no point in taking more than fits in the beam
        next_fringe = []
        for Y_t_n, p_t_n, extras_t_n, state_t_n, n in zip(Y_t, p_t, extras_t, state_t, fringe):
            Y_nll_t_n = -np.log(p_t_n[Y_t_n])

            for y_t_n, y_nll_t_n in zip(Y_t_n, Y_nll_t_n):
                n_new = Node(parent=n, state=state_t_n, value=y_t_n, cost=y_nll_t_n, extras=extras_t_n)
                next_fringe.append(n_new)

        #print('Len next_fringe before:', len(next_fringe))
        #next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost)
        #print('Len next_fringe after sort:', len(next_fringe))
        #next_fringe = next_fringe[:beam_width]
        #print('Len next_fringe after slice:', len(next_fringe))
        next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost)[:beam_width] # may move this into loop to save memory

    hypotheses.sort(key=lambda n: n.cum_cost)
    return hypotheses[:num_hypotheses]
