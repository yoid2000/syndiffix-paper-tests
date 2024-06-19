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
        # Higher _pcc_abs_weight_strength puts less weight on the absolute PI
        self._pcc_abs_weight_strength = 2.0
        # _pcc_abs_weight_max is the largest possible weight for absolute PI
        self._pcc_abs_weight_max = 0.5
        self._set_slide_amount()
        # _cov_adjust_min_intercept is the coverage value below which the
        # effective anonymity loss becomes zero
        self._cov_adjust_min_intercept = 1/10000
        # Higher _cov_adjust_strength leads to lower coverage adjustment
        self._cov_adjust_strength = 3.0

    def set_param(self, param, value):
        if param == 'pcc_abs_weight_strength':
            self._pcc_abs_weight_strength = value
            self._set_slide_amount()
        if param == 'pcc_abs_weight_max':
            self._pcc_abs_weight_max = value
            self._set_slide_amount()
        if param == 'cov_adjust_min_intercept':
            self._cov_adjust_min_intercept = value
        if param == 'cov_adjust_strength':
            self._cov_adjust_strength = value

    def get_param(self, param):
        if param == 'pcc_abs_weight_strength':
            return self._pcc_abs_weight_strength
        if param == 'pcc_abs_weight_max':
            return self._pcc_abs_weight_max
        if param == 'cov_adjust_min_intercept':
            return self._cov_adjust_min_intercept
        if param == 'cov_adjust_strength':
            return self._cov_adjust_strength
        return None

    def _set_slide_amount(self):
        '''
        0.01 = x * (1 - (pb ** strength))
        0.01 / x = 1 - (pb ** strength)
        (pb ** strength) = 1 - (0.01 / x)
        pb = ((1 - (0.01 / x)) ** (1/strength))
        '''
        # find the value of pb that would produce diff_weight of 0.5
        target_pb = ((1 - (0.01 / self._pcc_abs_weight_max)) ** (1/self._pcc_abs_weight_strength))
        self._pi_abs_slide_amount = 1 - target_pb

    def _get_pcc_abs_weight(self, pcc_base):
        pb = max(0, pcc_base - self._pi_abs_slide_amount)
        return 0.01 / max(0.01, (1 - (pb ** self._pcc_abs_weight_strength)))

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

    def _pcc(self, prec, cov):
        ''' Generates the precision-coverage-coefficient, PCC. prev is the precision
            of the attack, and cov is the coverage.
        '''
        cov_adj = self._cov_adjust(cov)
        return cov_adj * prec

    def _pcc_improve_absolute(self, pcc_base, pcc_attack):
        return pcc_attack - pcc_base

    def _pcc_improve_relative(self, pcc_base, pcc_attack):
        return (pcc_attack - pcc_base) / (1.00001 - pcc_base)

    def _pcc_improve(self, pcc_base, pcc_attack):
        pcc_rel = self._pcc_improve_relative(pcc_base, pcc_attack)
        pcc_abs = self._pcc_improve_absolute(pcc_base, pcc_attack)
        abs_weight = self._get_pcc_abs_weight(pcc_base)
        pcc_improve = (abs_weight * pcc_abs) + ((1-abs_weight) * pcc_rel)
        return pcc_improve

    def alscore(self, p_base, c_base, p_attack, c_attack):
        # Adjust the precision based on the coverage to make the
        # precision-coverage-coefficient pcc
        pcc_base = self._pcc(p_base, c_base)
        pcc_attack = self._pcc(p_attack, c_attack)
        return self._pcc_improve(pcc_base, pcc_attack)
