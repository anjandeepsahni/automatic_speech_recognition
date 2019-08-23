import numpy as np


# Node class for beam search.
class BeamNode(object):
    def __init__(self, parent, state, value, cost, extras):
        super(BeamNode, self).__init__()
        # value/predicted word for current node.
        self.value = value
        # parent Node, None for root
        self.parent = parent
        # current node's lstm hidden state
        self.state = state
        # cumulative cost of entire chain upto current node.
        self.cum_cost = parent.cum_cost + cost if parent else cost
        # length of entire chain
        self.length = 1 if parent is None else parent.length + 1
        # any extra variables to store for the node
        self.extras = extras
        # to hold the node's entire sequence.
        self._sequence = None

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


def beam_search(initial_state_function,
                generate_function,
                X,
                end_id,
                batch_size=1,
                beam_width=4,
                num_hypotheses=1,
                max_length=50,
                vocab_size=33):
    # initial_state_function: A function that takes X as input and returns
    #                         state (2-dimensonal numpy array with 1 row
    #                         representing decoder recurrent layer state).
    # generate_function: A function that takes Y_tm1 (1-dimensional numpy array
    #                    of token indices in decoder vocabulary generated at
    #                    previous step) and state_tm1 (2-dimensonal numpy array
    #                    of previous step decoder recurrent layer states) as
    #                    input and returns state_t (2-dimensonal numpy array of
    #                    current step decoder recurrent layer states),
    #                    p_t (2-dimensonal numpy array of decoder softmax
    #                    outputs) and optional extras (e.g. attention weights
    #                    at current step).
    # X: List of input token indices in encoder vocabulary.
    # end_id: Index of <end sequence> token in decoder vocabulary.
    # batch_size: Batch size. TBD !
    # beam_width: Beam size. Default 4. (NOTE: Fails for beam > vocab)
    # num_hypotheses: Number of hypotheses to generate. Default 1.
    # max_length: Length limit for generated sequence. Default 50.
    initial_state, initial_value = initial_state_function(X, batch_size)
    next_fringe = [BeamNode(parent=None,
                            state=initial_state,
                            value=initial_value,
                            cost=0.0,
                            extras=None)
                   ]
    hypotheses = []

    for step in range(max_length):
        fringe = []
        for n in next_fringe:
            if (step != 0 and n.value == end_id) or step == max_length - 1:
                hypotheses.append(n)
            else:
                fringe.append(n)

        if not fringe or len(hypotheses) >= num_hypotheses:
            # if not fringe:
            break

        Y_tm1 = [n.value for n in fringe]
        state_tm1 = [n.state for n in fringe]
        state_t, p_t, extras_t = generate_function(Y_tm1, state_tm1)
        Y_t = np.argsort(
            p_t, axis=1
        )[:, -beam_width:]  # no point in taking more than fits in the beam
        next_fringe = []
        for Y_t_n, p_t_n, extras_t_n, state_t_n, n in zip(
                Y_t, p_t, extras_t, state_t, fringe):
            Y_nll_t_n = -np.log(p_t_n[Y_t_n])

            for y_t_n, y_nll_t_n in zip(Y_t_n, Y_nll_t_n):
                n_new = BeamNode(parent=n,
                                 state=state_t_n,
                                 value=y_t_n,
                                 cost=y_nll_t_n,
                                 extras=extras_t_n)
                next_fringe.append(n_new)

        next_fringe = sorted(
            next_fringe, key=lambda n: n.cum_cost
        )[:beam_width]  # may move this into loop to save memory

    hypotheses.sort(key=lambda n: n.cum_cost)
    return hypotheses[:num_hypotheses]
