import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        n = y_probs.shape[1]
        prev_symbol = None

        for i in range(n):
            probs = y_probs[:, i, :]
            path_prob *= probs.max()
            i_sym = probs.argmax()

            if i_sym != blank:
                symbol = self.symbol_set[i_sym - 1]

                if symbol != prev_symbol:
                    decoded_path.append(symbol)

            prev_symbol = symbol

        decoded_path = ''.join(decoded_path)

        return decoded_path, path_prob
        

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
#         -----------

#         symbol_set [list[str]]:
#             all the symbols (the vocabulary without blank)

#         beam_width [int]:
#             beam width for selecting top-k hypotheses for expansion

#         """

        self.symbol_set = symbol_set
        self.beam_width = beam_width 

    def initialize_paths(self, y_probs):
        blank_path = {' ': y_probs[0, 0, 0]}
        symbol_paths = {self.symbol_set[i]: y_probs[i + 1, 0, 0] for i in range(len(self.symbol_set))}
        return blank_path, symbol_paths              
    

    def prune_paths(self, score_list, blank_path, symbol_path):
        sorted_paths = sorted(score_list, key=lambda x: x[1], reverse=True)
        cutoff = sorted_paths[:self.beam_width][-1][-1]

        blank_path = {k: v for k, v in blank_path.items() if v >= cutoff}
        symbol_path = {k: v for k, v in symbol_path.items() if v >= cutoff}
        return blank_path, symbol_path
    

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath = []

        bestPath, FinalPathScore = None, None

        blank_path, symbol_path = self.initialize_paths(y_probs)

        score_list = list(blank_path.items()) + list(symbol_path.items())

        for t in range(1, T):
            blank_path, symbol_path = self.prune_paths(score_list, blank_path, symbol_path)
            curr_prob_blank = y_probs[0, t, 0]
            curr_prob_symb = y_probs[1:, t, 0]

            # extend with blanks
            temp_blank_path = {}
            for k, v in blank_path.items():
                temp_blank_path[k] = v * curr_prob_blank

            for k, v in symbol_path.items():
                if k in temp_blank_path.keys():
                    temp_blank_path[k] += v * curr_prob_blank
                else:
                    temp_blank_path[k] = v * curr_prob_blank

            # extend with symbols
            
            temp_symbol_path = {}
            for idx, symb in enumerate(self.symbol_set):
                for k, v in blank_path.items():
                    if k == ' ':
                        new_key = symb
                    else:
                        new_key = k + symb

                    new_key = new_key.strip()
                    new_val = v * curr_prob_symb[idx]

                    if new_key not in temp_symbol_path:
                        temp_symbol_path[new_key] = new_val 
                    else:
                        temp_symbol_path[new_key] += new_val

                for k, v in symbol_path.items():
                    if k == ' ':
                        new_key = symb
                    elif k[-1] != symb:
                        new_key = k + symb
                    else:
                        new_key = k

                    new_key = new_key.strip()
                    new_val = v * curr_prob_symb[idx]

                    if new_key not in temp_symbol_path:
                        temp_symbol_path[new_key] = new_val 
                    else:
                        temp_symbol_path[new_key] += new_val

            blank_path = temp_blank_path.copy()
            symbol_path = temp_symbol_path.copy()

            score_list = list(blank_path.items()) + list(symbol_path.items())

        merged_path = blank_path.copy()
        for k, v in symbol_path.items():
            if k not in merged_path.keys():
                merged_path[k] = v
            else:
                merged_path[k] += v

        FinalPathScore = merged_path.copy()
        bestPath = max(merged_path, key=FinalPathScore.get)

        return bestPath, FinalPathScore
