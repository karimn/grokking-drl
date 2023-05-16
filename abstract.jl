abstract type AbstractModel end
abstract type AbstractValueModel <: AbstractModel end
abstract type AbstractPolicyModel <: AbstractModel end
abstract type AbstractActorCriticModel <: AbstractPolicyModel end
abstract type AbstractStrategy end
abstract type AbstractDRLAlgorithm end
abstract type AbstractBuffer end
abstract type AbstractLearner end
abstract type AbstractValueLearner <: AbstractLearner end
abstract type AbstractPolicyLearner <: AbstractLearner end
abstract type AbstractActorCriticLearner <: AbstractPolicyLearner end
