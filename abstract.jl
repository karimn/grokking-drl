abstract type AbstractAsyncEnv <: RLBase.AbstractEnv end
abstract type AbstractModel end
abstract type AbstractValueModel <: AbstractModel end
abstract type AbstractActionValueModel <: AbstractValueModel end
abstract type AbstractPolicyModel <: AbstractModel end
abstract type AbstractActorCriticModel <: AbstractPolicyModel end
abstract type AbstractStrategy end
abstract type AbstractDiscreteStrategy <: AbstractStrategy end
abstract type AbstractContinuousStrategy <: AbstractStrategy end
abstract type AbstractDRLAlgorithm end
abstract type AbstractBuffer end
abstract type AbstractLearner end
abstract type AbstractValueLearner <: AbstractLearner end
abstract type AbstractPolicyLearner <: AbstractLearner end
abstract type AbstractActorCriticLearner <: AbstractPolicyLearner end
