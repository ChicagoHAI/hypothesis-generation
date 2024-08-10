from ...register import Register

inference_register = Register("inference")

from .base import Inference
from .default import DefaultInference
from .filter_and_weight import FilterAndWeightInference
from .one_step_adaptive import OneStepAdaptiveInference
from .two_step_adaptive import TwoStepAdaptiveInference
from .upperbound import UpperboundInference
