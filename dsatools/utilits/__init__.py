from ._awgn import (awgn,
                    awgnDB,
                    wgn,
                    wgn_with_snr,
                    signal_like_noise)

from ._auxiliary import (pad_to_power_of_2,
                         pad_noises)

from ._types import(fixpoint,
                    is_1d,
                    is_complex,
                    to_1d,
                    to_2d, 
                    to_callback,
                    as1darray)

from ._probe import probe, probe_filter

from ._findpeaks import findpeaks

from ._barycenter import barycenter

from ._math import(cexp,
                   cexp2pi,
                   polyval,
                   gamma, 
                   join_subsets, 
                   cross_subsets,
                   corcof,
                   standartize,
                   normalize)

from ._geometry import (calc_line,
                        flip_vector,
                        ols_line,
                        piecewise_ols_line,
                        point_to_vector_distance)


from ._smooth import moving_average, movav, movstd
from ..operators import *
