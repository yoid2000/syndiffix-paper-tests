import numpy as np

class ALScore:
    '''
    ALScore is used to generate an anonymity loss score (ALS). The max
    ALS is 1.0, which corresponds to complete anonymity loss, and is equivalent
    to publishing the original data. An ALS of 0.0 means that the there is
    no anonymity loss. What this means in practice is that the quality of
    attribute inferences about individuals in the synthetic dataset is
    statistically equivalent to the quality of attribute inferences made
    from a non-anonymized dataset about individuals that are not in that dataset.
    The ALS can be negative.  An ALS of 0.5 can be regarded conservatively as a
    safe amount of loss. In other words, the loss is little enough that it 
    eliminates attacker incentive.
    '''
    def __init__(self):
        # _pcc_abs_weight is the weight given to the absolute PCC difference
        self._pcc_abs_weight = 0.5
        # _cov_adjust_min_intercept is the coverage value below which the
        # effective anonymity loss becomes zero
        self._cov_adjust_min_intercept = 1/10000
        # Higher _cov_adjust_strength leads to lower coverage adjustment
        self._cov_adjust_strength = 3.0

    def set_param(self, param, value):
        if param == 'pcc_abs_weight':
            self._pcc_abs_weight = value
        if param == 'cov_adjust_min_intercept':
            self._cov_adjust_min_intercept = value
        if param == 'cov_adjust_strength':
            self._cov_adjust_strength = value

    def get_param(self, param):
        if param == 'pcc_abs_weight':
            return self._pcc_abs_weight
        if param == 'cov_adjust_min_intercept':
            return self._cov_adjust_min_intercept
        if param == 'cov_adjust_strength':
            return self._cov_adjust_strength
        return None

    def _underlying_prec_cov_curve(self):
        pi1 = 1.0    # PI intercept at low coverage
        cov2, pi2 = 1.0, 0.0    # PI and coverage intercept at high coverage
        m = (pi2 - pi1) / (np.log10(cov2) - np.log10(self._cov_adjust_min_intercept))
        b = pi1 - m * np.log10(self._cov_adjust_min_intercept)
        return m, b

    def _cov_adjust(self, cov):
        m, b = self._underlying_prec_cov_curve()
        adjust = (m * np.log10(cov) + b) ** self._cov_adjust_strength
        # Note: reverse of this is:
        # COV = 10 ** ((PI ** (1/self._cov_adjust_strength) - b) / m)
        return 1 - adjust

    def _pcc_improve_absolute(self, pcc_base, pcc_attack):
        return pcc_attack - pcc_base

    def _pcc_improve_relative(self, pcc_base, pcc_attack):
        return (pcc_attack - pcc_base) / (1.00001 - pcc_base)

    def _pcc_improve(self, pcc_base, pcc_attack):
        pcc_rel = self._pcc_improve_relative(pcc_base, pcc_attack)
        pcc_abs = self._pcc_improve_absolute(pcc_base, pcc_attack)
        pcc_improve = (self._pcc_abs_weight * pcc_abs) + ((1-self._pcc_abs_weight) * pcc_rel)
        return pcc_improve

    def pcc(self, prec, cov):
        ''' Generates the precision-coverage-coefficient, PCC. prev is the precision
            of the attack, and cov is the coverage.
        '''
        cov_adj = self._cov_adjust(cov)
        return cov_adj * prec

    def alscore(self,
                p_base = None,
                c_base = None,
                p_attack = None,
                c_attack = None,
                pcc_base = None,
                pcc_attack = None
                ):
        ''' alscore can be called with either p_x and c_x, or pcc_x
        '''
        if pcc_base is None and p_base is not None and c_base is not None:
            # Adjust the precision based on the coverage to make the
            # precision-coverage-coefficient pcc
            pcc_base = self.pcc(p_base, c_base)
        if pcc_attack is None and p_attack is not None and c_attack is not None:
            pcc_attack = self.pcc(p_attack, c_attack)
        if pcc_base is not None and pcc_attack is not None:
            return self._pcc_improve(pcc_base, pcc_attack)
        return None
