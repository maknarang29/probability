"""The Poisson Binomial distribution class."""
import tensorflow.compat.v2 as tf
import numpy as np
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python import distributions
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
import pyfftw

class PoissonBinomial(distribution.Distribution):

    def __init__(self,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
             name='PoissonBinomial'):
        self._dtype = dtype_util.common_dtype([probs], tf.float32)
        self._probs = tensor_util.convert_nonref_to_tensor(
        probs, dtype=self._dtype)
        self._total_count = tf.shape(self._probs)
        self._pmf_list = self.get_pmf()
        self._cdf_list = tf.math.cumsum(self._pmf_list)

    def samples_n(n,probs,seed=None):
        np.random.seed(seed)
        res = np.zeros(n)
        i = 0
        for j in probs:
          for i in range(n):
              res[i] += np.array(distributions.Binomial(total_count = 1,probs = j).sample(1))[0]
        return(res)
    def ppb_generic(pmf):
      return np.cumsum(pmf)
    
    def dpb_conv(obs,probs):
      size = len(probs)
      res = np.zeros(size+1)
      res[0] = 1-probs[0]
      res[1] = probs[0]
      for i in range(1,size):
        if(probs[i]):
          for j in range(i,-1,-1):
            if (res[j]):
              res[j+1] += res[j]*probs[i]
              res[j] *= 1 - probs[i]
      res /= np.sum(res)
      return res
    
    def ppb_conv(obs,probs):
      size = probs(len)
      pmf = dpb_conv(obs,probs)
      res = ppb_generic(pmf)
      return (res)

    def fft_probs(probs_A, probs_B):
      sizeA = len(probsA)
      sizeB = len(probsB)
      sizeRes = sizeA + sizeB -1

      res = np.empty(sizeRes,dtype = 'float64')

      probsA_fft = pyfftw.empty_aligned(sizeRes)
      probsB_fft = pyfftw.empty_aligned(sizeRes)
      result_fft = pyfftw.empty_aligned(sizeRes)

      padded_probsA = pyfftw.zeros_aligned(sizeRes)
      padded_probsA[:sizeA] = probs_A
      fftw_planA = pyfftw(padded_probsA,probsA_fft,flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
      fftw_planA.execute()

      padded_probsB = pyfftw.zeros_aligned(sizeRes)
      padded_probsB[:sizeB] = probs_B
      fftw_planB = pyfftw(padded_probsB,probsB_fft,flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
      fftw_planB.execute()

      for i in range(sizeRes):
          result_fft[i].real = (probsA_fft[i].real*probsB_fft[i].real - probsA_fft[i].imag*probsB_fft[i].imag)/sizeRes
          result_fft[i].imag = (probsA_fft[i].real*probsB_fft[i].imag + probsA_fft[i].imag*probsB_fft[i].real)/sizeRes

      planResult = pyfftw.FFTW(result_fft, res, direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'))
      planResult.execute()

      return (res)
