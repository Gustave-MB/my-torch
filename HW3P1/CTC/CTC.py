import numpy as np

class CTC(object):

    def __init__(self, BLANK=0):
        """
        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with a blank.

        Input
        -----
        target: (np.array, dim=(target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim=(2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim=(2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]

        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)
        skip_connect = []

        for i in range(N):
            if i + 2 <= N - 1:
                skip_connect.append(1) if extended_symbols[i] != self.BLANK and extended_symbols[i] != extended_symbols[i + 2] else skip_connect.append(0)
            else:
                skip_connect.append(0)

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect[::-1]).reshape((N,))

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities
        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        alpha[0][0] = logits[0, extended_symbols[0]] # TODO: Intialize alpha[0][0]
        alpha[0][1] = logits[0, extended_symbols[1]]# TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        for time in range(1, T, 1):
                for sym in range(S):
                    if sym == 0:
                        alpha[time][sym] = alpha[time-1][sym]

                    elif extended_symbols[sym] == 0:
                        alpha[time][sym] = sum(alpha[time-1][sym - 1 : sym + 1])
                
                    else:
                        if sym < 2:
                            alpha[time][sym] = sum(alpha[time-1][sym - 1 : sym + 1])
                        else:
                            if skip_connect[sym] == 1:
                                alpha[time][sym] = sum(alpha[time-1][sym - 2 : sym + 1]) 
                            elif skip_connect[sym - 2] == 1:
                                if skip_connect[sym - 1] == 1:
                                    alpha[time][sym] = sum(alpha[time-1][sym]) 
                                else:
                                    alpha[time][sym] = sum(alpha[time-1][sym - 1 : sym + 1])
                            else:
                                alpha[time][sym] = sum(alpha[time-1][sym - 2 : sym + 1]) 
                
                    alpha[time][sym] *= logits[time, extended_symbols[sym]]
        # IMP: Remember to check for skipConnect when calculating alpha
        self.alpha = alpha
        return alpha
        # <---------------------------------------------
        

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))
        

        # -------------------------------------------->
        betahat = np.zeros(shape=(T, S))
        skip_connect_back = list(skip_connect)
        skip_connect_back.reverse()

        betahat[-1][-1] = logits[-1][extended_symbols[-1]]
        betahat[-1][-2] = logits[-1][extended_symbols[-2]] 

        for t in reversed(range(T-1)):
            betahat[t][-1] = betahat[t+1][-1]*logits[t][extended_symbols[-1]]
            for s in reversed(range(S-1)):
                betahat[t][s] = betahat[t+1][s] + betahat[t+1][s+1]
                if skip_connect_back[s] == 1:
                    betahat[t][s]+=betahat[t+1][s+2]
                betahat[t][s]= betahat[t][s] * logits[t][extended_symbols[s]]
        
        
        for t in reversed(range(T)):
            for s in reversed(range(S)):
                if logits[t][extended_symbols[s]]!=0:
                    beta[t][s] = betahat[t][s] / logits[t][extended_symbols[s]]       
        # <--------------------------------------------

        return beta


    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        for t in range(T):
            sumgamma[t] = 0
            for s in range(S):
                gamma[t][s] = alpha[t][s] * beta[t][s]
                sumgamma[t]+=gamma[t][s]
            for s in range(S): gamma[t][s] = gamma[t][s] / sumgamma[t]
        # <---------------------------------------------

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior probabilities, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            targets_comp = target[batch_itr, :target_lengths[batch_itr]]
            logits_comp = logits[:input_lengths[batch_itr], batch_itr]
            self.extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target=targets_comp)
            alphas = self.ctc.get_forward_probs(logits_comp, self.extended_symbols, skip_connect)
            betas = self.ctc.get_backward_probs(logits_comp, self.extended_symbols, skip_connect)
            gammas = self.ctc.get_posterior_probs(alphas, betas)
            for i in range(logits_comp.shape[0]):
                for ix, symbol in enumerate(self.extended_symbols):
                    total_loss[batch_itr] -= (np.log(logits_comp[i][symbol])*gammas[i][ix]) 

            #     Take an average over all batches and return final result
            # <---------------------------------------------

        total_loss = np.sum(total_loss) / B
        self.total_loss = total_loss
		
        return total_loss
         

    def backward(self):
        """

        CTC loss backward

        Calculate the gradients w.r.t the parameters and return the derivative
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(Symbols)]:
        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
        derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        
        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for a single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute the derivative of divergence and store them in dY
            # <---------------------------------------------
            # 
            # -------------------------------------------->
            # Your code for CTC derivative computation goes here
            # Truncate the target and logits to target length and input length respectively
            targets_comp = self.target[batch_itr, :self.target_lengths[batch_itr]]
            logits_comp = self.logits[:self.input_lengths[batch_itr], batch_itr]

            #     Extend target sequence with blank
            self.extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target=targets_comp)
            alphas = self.ctc.get_forward_probs(logits_comp, self.extended_symbols, skip_connect)
            betas = self.ctc.get_backward_probs(logits_comp, self.extended_symbols, skip_connect)
            gammas = self.ctc.get_posterior_probs(alphas, betas)

            #     Compute derivative of divergence and store them in dY
            for seq in range(logits_comp.shape[0]):
                for ix, symbol in enumerate(self.extended_symbols):
                    dY[seq, batch_itr, symbol] -= gammas[seq][ix]/logits_comp[seq][symbol]

        return dY
           


