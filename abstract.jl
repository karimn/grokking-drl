abstract type AbstractModel end
abstract type AbstractValueBasedModel <: AbstractModel end
abstract type AbstractPolicyBasedModel <: AbstractModel end
abstract type AbstractStrategy end
abstract type AbstractDRLAlgorithm end
abstract type AbstractBuffer end
abstract type AbstractLearner end
abstract type AbstractValueLearner <: AbstractLearner end
abstract type AbstractPolicyLearner <: AbstractLearner end
